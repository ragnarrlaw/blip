import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

from config.conf import Config
from model.model import Assignment, MasterQuestion, Grade

log = logging.getLogger("blip.assignment_service")


def clean_text(t: str) -> str:
    """Strips whitespace and punctuation for robust text matching."""
    if not t: return ""
    return "".join(c.lower() for c in t if c.isalnum())


def get_or_create_assignment(db: Session, course_id: int, cmid: int, assignment_name: str) -> Assignment:
    item_key = f"quiz:{cmid}"
    assignment = db.query(Assignment).filter(Assignment.item_key == item_key).first()
    if not assignment:
        assignment = Assignment(
            course_id=course_id,
            name=assignment_name,
            item_key=item_key,
            grading_status="pending"
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
    return assignment


def _seed_grade(db: Session, assignment_id: int, student_id: str, mq: MasterQuestion, student_answer: str):
    exists = db.query(Grade).filter(
        Grade.assignment_id == assignment_id,
        Grade.student_id == student_id,
        Grade.question_id == mq.canonical_id
    ).first()

    if not exists:
        db.add(Grade(
            assignment_id=assignment_id,
            student_id=student_id,
            question_id=mq.canonical_id,
            question_type=mq.question_type,
            question_text=mq.question_text,
            scenario_text=mq.scenario_text or "",
            rubric=mq.rubric,
            model_answer=mq.model_answer,
            student_answer=student_answer,
            is_answered=len(student_answer.strip()) > 1 and student_answer.strip() != "-",
            max_mark=mq.max_mark,
            mark_assigned=0.0,
            confidence_score=0.0
        ))
        db.commit()


def ingest_moodle_submissions(
        db: Session, config: Config, course_id: int, cmid: int, assignment_name: str,
        raw_data: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    assignment = get_or_create_assignment(db, course_id, cmid, assignment_name)

    # Flatten Moodle export
    student_records = raw_data[0] if raw_data and isinstance(raw_data[0], list) else raw_data

    master_questions = db.query(MasterQuestion).filter(MasterQuestion.assignment_id == assignment.id).all()
    if not master_questions:
        raise ValueError("No Master Questions found. Upload the exam rubric first.")

    # Sort master questions into Independent vs Scenario
    independent_mqs = [mq for mq in master_questions if not mq.is_scenario_based]
    scenario_mqs = [mq for mq in master_questions if mq.is_scenario_based]

    # Group scenario questions by their shared scenario_text
    scenario_groups = {}
    for mq in scenario_mqs:
        c_text = clean_text(mq.scenario_text)
        if c_text not in scenario_groups:
            scenario_groups[c_text] = []
        scenario_groups[c_text].append(mq)

    client = genai.Client(api_key=config.gemini_api_key)
    total_seeded = 0
    students_processed = 0

    for row in student_records:
        student_id = str(row.get("idnumber") or row.get("username") or "").strip()
        if not student_id:
            continue

        students_processed += 1

        for key, value in row.items():
            key_lower = str(key).lower()
            if key_lower.startswith("response"):
                q_num = key_lower.replace("response", "")
                q_text = str(row.get(f"question{q_num}", ""))
                r_text = str(value).strip() if value else "-"

                if not q_text:
                    continue

                c_q_text = clean_text(q_text)

                # 1. Match Independent Questions (Bypass LLM)
                matched_indep = next((mq for mq in independent_mqs if
                                      clean_text(mq.question_text) in c_q_text or c_q_text in clean_text(
                                          mq.question_text)), None)
                if matched_indep:
                    _seed_grade(db, assignment.id, student_id, matched_indep, r_text)
                    total_seeded += 1
                    continue

                # 2. Match Scenario Questions (Send to LLM for segmentation)
                matched_scenario_group = next((group for c_scenario, group in scenario_groups.items() if
                                               c_scenario in c_q_text or c_q_text in c_scenario), None)

                if matched_scenario_group and r_text != "-":
                    sub_questions = [{"canonical_id": mq.canonical_id, "text": mq.question_text} for mq in
                                     matched_scenario_group]

                    prompt = f"""
                    You are an academic parser. A student submitted a single block of text responding to a scenario with multiple sub-questions.
                    Extract the student's answer for EACH sub-question.
                    Return a valid JSON array of objects with keys: "canonical_id" and "student_answer". If left blank, use "-".

                    Sub-Questions to find:
                    {json.dumps(sub_questions, indent=2)}

                    Student's Raw Monolithic Answer:
                    {r_text}
                    """
                    try:
                        resp = client.models.generate_content(
                            model=config.gemini_model_for_question_extraction,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                response_mime_type="application/json"
                            )
                        )
                        parsed_segments = json.loads(resp.text)

                        for seg in parsed_segments:
                            target_mq = next(
                                (mq for mq in matched_scenario_group if mq.canonical_id == seg.get("canonical_id")),
                                None)
                            if target_mq:
                                _seed_grade(db, assignment.id, student_id, target_mq, seg.get("student_answer", "-"))
                                total_seeded += 1

                    except Exception as e:
                        log.error(f"LLM segmentation failed for {student_id}: {e}")
                        # Fallback: Save the monolithic block to the first sub-question so data isn't lost
                        _seed_grade(db, assignment.id, student_id, matched_scenario_group[0], r_text)
                        total_seeded += 1

    return {"assignment_id": assignment.id, "students_processed": students_processed, "jobs_seeded": total_seeded}


def ingest_research_dataset(
        db: Session, course_id: int, cmid: int, assignment_name: str, raw_data: Dict[str, Any]
) -> Dict[str, Any]:
    assignment = get_or_create_assignment(db, course_id, cmid, assignment_name)
    dataset = raw_data.get("dataset", [])

    for item in dataset:
        exists = db.query(Grade).filter(
            Grade.assignment_id == assignment.id, Grade.student_id == str(item["student_id"]),
            Grade.question_id == item["question_id"]
        ).first()

        if not exists:
            db.add(Grade(
                assignment_id=assignment.id, student_id=str(item["student_id"]), question_id=item["question_id"],
                question_type=item.get("question_type", "independent"), question_text=item["question_text"],
                scenario_text=item.get("scenario_text", ""), rubric=item["rubric"], model_answer=item["model_answer"],
                student_answer=item["student_answer"], is_answered=item["is_answered"], max_mark=item["max_mark"],
                mark_assigned=0.0, confidence_score=0.0
            ))
    db.commit()
    return {"assignment_id": assignment.id, "jobs_seeded": len(dataset), "type": "direct_dataset_ingest"}
