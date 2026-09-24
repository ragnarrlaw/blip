import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
from sqlalchemy.orm import Session

from config.conf import Config
from model.model import Assignment, MasterQuestion
from services.submission_parser import batch_parse_and_seed_submissions

log = logging.getLogger("blip.assignment_service")

IDENTITY_KEYS = ("idnumber", "username", "emailaddress", "email")


def get_or_create_assignment(
    db: Session, course_id: int, cmid: int, assignment_name: str
) -> Assignment:
    item_key = f"quiz:{cmid}"
    assignment = db.query(Assignment).filter(Assignment.item_key == item_key).first()
    if not assignment:
        assignment = Assignment(
            course_id=course_id,
            name=assignment_name,
            item_key=item_key,
            grading_status="pending",
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)
    return assignment


def ingest_moodle_submissions(
    db: Session,
    config: Config,
    course_id: int,
    cmid: int,
    assignment_name: str,
    raw_data: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    batch_size: int = 15,
) -> Dict[str, Any]:
    """
    Ingests Moodle quiz JSON data, extracts single-textbox submissions,
    and seeds the Grade table using LLM-assisted segmentation.
    """
    assignment = get_or_create_assignment(db, course_id, cmid, assignment_name)

    # Flatten nested lists if Moodle export contains batched groups
    if raw_data and isinstance(raw_data[0], list):
        flattened_rows = [item for sublist in raw_data for item in sublist]
    else:
        flattened_rows = raw_data

    # Extract student identities and aggregate single-textbox answers
    raw_submissions: Dict[str, str] = {}
    for row in flattened_rows:
        student_id = next(
            (str(row.get(k)).strip() for k in IDENTITY_KEYS if row.get(k)), None
        )
        if not student_id:
            continue

        response_texts = [
            str(v).strip()
            for k, v in row.items()
            if str(k).lower().startswith("response") and v
        ]
        if response_texts:
            raw_submissions[student_id] = "\n\n".join(response_texts)

    if not raw_submissions:
        raise ValueError(
            "No valid student submissions with identifiable IDs and responses found in data."
        )

    master_questions = (
        db.query(MasterQuestion).filter(MasterQuestion.course_id == course_id).all()
    )
    if not master_questions:
        raise ValueError(
            f"No Master Questions found for Course ID {course_id}. Upload master questions first."
        )

    # Partition submissions to stay within LLM output token limits
    items = list(raw_submissions.items())
    total_seeded = 0
    batches_dispatched = 0

    for i in range(0, len(items), batch_size):
        batch_dict = dict(items[i : i + batch_size])
        seeded = batch_parse_and_seed_submissions(
            db=db,
            config=config,
            assignment_id=assignment.id,
            master_questions=master_questions,
            raw_submissions=batch_dict,
        )
        total_seeded += seeded
        batches_dispatched += 1

    return {
        "assignment_id": assignment.id,
        "assignment_name": assignment.name,
        "total_students_found": len(raw_submissions),
        "total_grade_jobs_seeded": total_seeded,
        "batches_processed": batches_dispatched,
    }


def ingest_submissions_from_file(
    db: Session,
    config: Config,
    course_id: int,
    cmid: int,
    assignment_name: str,
    file_path: Path,
    batch_size: int = 15,
) -> Dict[str, Any]:
    """Helper to ingest JSON submissions directly from a local file path."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ingest_moodle_submissions(
        db, config, course_id, cmid, assignment_name, data, batch_size
    )

