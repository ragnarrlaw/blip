import logging
from typing import Dict, List
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

from config.conf import Config
from model.model import MasterQuestion, Grade
from schema.schema import BatchParsedSubmissions

log = logging.getLogger("blip.submission_parser")

SYSTEM_PROMPT = """
You are a precision data-extraction assistant. 
Students were forced to answer multiple sub-questions inside a single large text box. 
Your task is to read each raw student submission and segment their text, matching each paragraph or section to the correct Master Question ID.
Do not grade the answers. Do not alter the student's text. Simply extract and map the text boundaries.
"""


def batch_parse_and_seed_submissions(
        db: Session,
        config: Config,
        assignment_id: int,
        master_questions: List[MasterQuestion],
        raw_submissions: Dict[str, str]  # Dictionary of {real_student_id: raw_text_block}
) -> int:
    """
    Batches raw Moodle submissions, uses Gemini to segment them by canonical_id,
    and seeds the Grade table for the grading worker queue.
    """
    # 1. Ephemeral ID Mapping (Anonymization Layer)
    ephemeral_map = {}
    ephemeral_submissions = []

    for i, (real_student_id, raw_text) in enumerate(raw_submissions.items(), 1):
        e_id = f"RAW_{i:03d}"
        ephemeral_map[e_id] = real_student_id
        ephemeral_submissions.append(f"--- Submission ID: {e_id} ---\n{raw_text}\n")

    # 2. Build the context for the LLM
    mq_context = "\n".join([
        f"- ID: {q.canonical_id} | Question: {q.question_text}"
        for q in master_questions
    ])

    prompt = (
            f"Master Questions for this assignment:\n{mq_context}\n\n"
            f"Raw Student Submissions:\n" + "\n".join(ephemeral_submissions)
    )

    # 3. Invoke LLM for structural extraction
    log.info(f"Dispatching batch of {len(raw_submissions)} raw submissions for segmentation.")
    client = genai.Client(api_key=config.gemini_api_key)

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=BatchParsedSubmissions,
        system_instruction=SYSTEM_PROMPT
    )

    response: types.GenerateContentResponse = client.models.generate_content(
        model=config.gemini_model_name,  # gemini-1.5-flash is ideal here
        contents=prompt,
        config=generation_config,
    )

    if not response.parsed:
        raise RuntimeError("LLM failed to segment submissions into the required JSON schema.")

    parsed_batch: BatchParsedSubmissions = response.parsed

    # 4. Map back to real IDs and seed the Grade table
    db_grades = []
    mq_lookup = {q.canonical_id: q for q in master_questions}

    for sub in parsed_batch.submissions:
        real_student_id = ephemeral_map.get(sub.ephemeral_id)
        if not real_student_id:
            continue

        for ans in sub.answers:
            master_q = mq_lookup.get(ans.canonical_id)
            if not master_q:
                continue  # Skip if the LLM hallucinated an ID

            new_grade_job = Grade(
                assignment_id=assignment_id,
                student_id=real_student_id,
                question_id=master_q.canonical_id,
                question_type=master_q.question_type,
                question_text=master_q.question_text,
                scenario_text=master_q.scenario_text or "",
                rubric=master_q.rubric,
                model_answer=master_q.model_answer,
                student_answer="" if ans.is_blank else ans.student_answer,
                is_answered=not ans.is_blank,
                max_mark=master_q.max_mark,
                mark_assigned=0.0,  # Pending grading worker
                confidence_score=0.0,  # Pending grading worker
            )
            db_grades.append(new_grade_job)

    db.add_all(db_grades)
    db.commit()

    log.info(f"Seeded {len(db_grades)} individual Grade jobs into the queue.")
    return len(db_grades)