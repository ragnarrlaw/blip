import logging
from pathlib import Path
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

from config.conf import Config
from model.model import (MasterQuestion)
from schema.schema import MasterQuestionList
from context.parser import parse, sanitize_text

log = logging.getLogger("blip.master_questions")

SYSTEM_PROMPT = """
You are an expert academic curriculum designer. Your task is to process a raw exam paper, syllabus, or assignment document and extract every question into a highly structured format.

For each question found:
1. Determine if it is an 'independent' question (self-contained) or a 'scenario' question (relies on a shared background story or dataset).
2. Extract the exact question text.
3. Extract or formulate the shared scenario text (if applicable). Do not repeat the scenario text in the question text.
4. Extract the allocated maximum marks.
5. Extract or formulate a comprehensive model answer.
6. Extract or formulate a detailed grading rubric defining exactly how marks are awarded. Use markdown bullets.
"""


def generate_master_questions_from_file(
        db: Session,
        config: Config,
        course_id: int,
        file_path: Path
) -> int:
    """
    Parses an uploaded document, uses Gemini to extract structured questions,
    and saves them to the MasterQuestion table.
    """
    # 1. Read and parse the raw file (reusing your Docling setup)
    log.info(f"Extracting raw text from {file_path.name}")
    file_bytes = file_path.read_bytes()
    parsed_doc = parse(file_path.name, file_bytes)
    clean_text = sanitize_text(parsed_doc.text)

    # 2. Invoke Gemini with strict JSON schema enforcement
    log.info("Sending document to LLM for structured extraction...")
    client = genai.Client(api_key=config.gemini_api_key)

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=MasterQuestionList,
        system_instruction=SYSTEM_PROMPT
    )

    prompt = f"Extract the questions, model answers, and rubrics from the following academic document:\n\n{clean_text}"

    response = client.models.generate_content(
        model=config.gemini_model_name,  # Usually gemini-1.5-flash for fast, cheap extraction
        contents=prompt,
        config=generation_config,
    )

    if not response.parsed:
        raise RuntimeError("LLM failed to return valid JSON matching the MasterQuestionList schema.")

    extracted_data: MasterQuestionList = response.parsed

    # 3. Bulk insert into the database
    log.info(f"Successfully extracted {len(extracted_data.questions)} master questions. Saving to database.")

    db_questions = []
    for q in extracted_data.questions:
        db_q = MasterQuestion(
            course_id=course_id,
            canonical_id=q.canonical_id,
            question_text=q.question_text,
            question_type=q.question_type,
            scenario_text=q.scenario_text,
            max_mark=q.max_mark,
            model_answer=q.model_answer,
            rubric=q.rubric
        )
        db_questions.append(db_q)

    # Note: Depending on your UI flow, you might return the `extracted_data` here
    # for the lecturer to review in the browser *before* committing to the DB.
    db.add_all(db_questions)
    db.commit()

    return len(db_questions)