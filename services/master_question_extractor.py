import logging
from pathlib import Path
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

from config.conf import Config
from model.model import MasterQuestion
from schema.schema import MasterQuestionList

log = logging.getLogger("blip.master_question_extractor")

SYSTEM_PROMPT = """
You are an expert academic parser. Your task is to extract structured Master Questions 
from the provided exam paper/rubric text.
Identify the canonical ID (e.g., 'Q1', 'Q3(a)'), the question text, the max mark, 
the model answer, and the grading rubric.
If a question relies on a shared scenario, extract the scenario text and set the question type to 'scenario'.
"""

def generate_master_questions_from_file(
        db: Session, config: Config, assignment_id: int, file_path: Path
) -> int:
    """
    Parses a raw text exam paper using Gemini and seeds the MasterQuestion table.
    """
    raw_text = file_path.read_text(encoding="utf-8")

    client = genai.Client(api_key=config.gemini_model_for_question_extraction)

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=MasterQuestionList,
        system_instruction=SYSTEM_PROMPT,
    )

    log.info(f"Dispatching exam paper to LLM for assignment_id {assignment_id}")

    response: types.GenerateContentResponse = client.models.generate_content(
        model=config.gemini_model_name,
        contents=raw_text,
        config=generation_config,
    )

    if not response.parsed:
        raise RuntimeError("LLM failed to parse the exam paper into the required schema.")

    parsed_paper: MasterQuestionList = response.parsed

    db_questions = []
    for q in parsed_paper.questions:
        # Check to prevent duplicate uploads
        exists = db.query(MasterQuestion).filter(
            MasterQuestion.assignment_id == assignment_id,
            MasterQuestion.canonical_id == q.canonical_id
        ).first()

        if not exists:
            db_questions.append(
                MasterQuestion(
                    assignment_id=assignment_id,
                    canonical_id=q.canonical_id,
                    question_text=q.question_text,
                    question_type=q.question_type,
                    scenario_text=q.scenario_text,
                    is_scenario_based=(q.question_type == "scenario"),
                    max_mark=q.max_mark,
                    model_answer=q.model_answer,
                    rubric=q.rubric,
                )
            )

    if db_questions:
        db.add_all(db_questions)
        db.commit()

    return len(db_questions)
