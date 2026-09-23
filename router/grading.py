# from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
# from sqlalchemy.orm import Session
# from typing import List, Optional
#
# from db.db import db_manager
# from config.conf import get_config, Config
# from model.model import Grade
# from services.grading_service import GradingService
#
# router = APIRouter(prefix="/grading", tags=["Grading"])
#
# def get_grading_service(cfg: Config = Depends(get_config)) -> GradingService:
#     return GradingService(cfg)
#
# @router.post("/batch")
# def dispatch_batch_grading(
#     question_id: str,
#     course_slug: str,
#     assignment_id: int,
#     use_rag: bool = True,
#     batch_size: int = 15,
#     background_tasks: BackgroundTasks = BackgroundTasks(),
#     db: Session = Depends(db_manager.get_session),
#     grader: GradingService = Depends(get_grading_service)
# ):
#     """
#     Pulls ungraded student answers for a specific question, partitions them
#     into dynamic batches, and runs evaluation as a background worker.
#     """
#     ungraded = db.query(Grade.id).filter(
#         Grade.assignment_id == assignment_id,
#         Grade.question_id == question_id,
#         Grade.confidence_score == 0.0 # Pending status indicator
#     ).all()
#
#     grade_ids = [g[0] for g in ungraded]
#     if not grade_ids:
#         return {"message": "No pending submissions to grade for this question."}
#
#     # Partition grade IDs into manageable batches
#     partitions = [grade_ids[i:i + batch_size] for i in range(0, len(grade_ids), batch_size)]
#
#     for chunk in partitions:
#         background_tasks.add_task(
#             grader.grade_batch,
#             db=db,
#             grade_ids=chunk,
#             course_slug=course_slug,
#             use_rag=use_rag
#         )
#
#     return {
#         "status": "queued",
#         "total_jobs": len(grade_ids),
#         "batches_dispatched": len(partitions)
#     }
#
# @router.post("/single/{grade_id}")
# def dispatch_single_grading(
#     grade_id: int,
#     course_slug: str,
#     use_rag: bool = True,
#     db: Session = Depends(db_manager.get_session),
#     grader: GradingService = Depends(get_grading_service)
# ):
#     """Evaluates an isolated submission for step-by-step review or debugging."""
#     result = grader.grade_single(db=db, grade_id=grade_id, course_slug=course_slug, use_rag=use_rag)
#     if not result:
#         raise HTTPException(status_code=404, detail="Grade job failed or not found.")
#     return {
#         "id": result.id,
#         "mark": result.mark_assigned,
#         "confidence": result.confidence_score,
#         "justifications": result.justifications,
#         "criterion_scores": result.criterion_scores
#     }

import json
from pathlib import Path
from fastapi import APIRouter, Depends, BackgroundTasks, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session

from db.db import db_manager
from config.conf import get_config, Config
from model.model import MasterQuestion
from services.grading_service import GradingService
from services.master_question_extractor import generate_master_questions_from_file
from services.submission_parser import batch_parse_and_seed_submissions
from services.ingestion import ingest_course_file

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def get_grading_service(cfg: Config = Depends(get_config)) -> GradingService:
    return GradingService(cfg)


@router.post("/1-ingest-content")
def api_ingest_content(
        course_id: int = Form(...),
        item_key: str = Form(...),
        content_kind: str = Form("material"),
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        db: Session = Depends(db_manager.get_session),
        config: Config = Depends(get_config)
):
    """Step 1: Upload course materials for RAG vectorization."""
    file_bytes = file.file.read()
    version = ingest_course_file(
        db, config, course_id, item_key, file.filename,
        content_kind, file_bytes, background_tasks
    )
    return {"message": f"Ingested {file.filename} as version {version.version}. Vectorization queued."}


@router.post("/2-ingest-master-questions")
def api_ingest_master_questions(
        course_id: int = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(db_manager.get_session),
        config: Config = Depends(get_config)
):
    """Step 2: Extract structured Master Questions from an uploaded syllabus/exam paper."""
    temp_path = Path(config.blip_output_dir) / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(file.file.read())

    try:
        count = generate_master_questions_from_file(db, config, course_id, temp_path)
        return {"message": f"Successfully extracted and saved {count} master questions."}
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/3-ingest-submissions")
def api_ingest_submissions(
        assignment_id: int = Form(...),
        course_id: int = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(db_manager.get_session),
        config: Config = Depends(get_config)
):
    """Step 3: Upload raw Moodle JSON, segment answers, and seed the Grade queue."""
    raw_json = json.loads(file.file.read())
    master_qs = db.query(MasterQuestion).filter(MasterQuestion.course_id == course_id).all()

    if not master_qs:
        raise HTTPException(status_code=400, detail="No master questions found for this course.")

    # Convert Moodle export format to {student_id: raw_text}
    # (Assuming the raw_json is a flat dict of {id: text} for this demo)
    raw_submissions = raw_json

    job_count = batch_parse_and_seed_submissions(db, config, assignment_id, master_qs, raw_submissions)
    return {"message": f"Segmented submissions and seeded {job_count} grading jobs."}


@router.post("/4-execute-grading")
def api_execute_grading(
        assignment_id: int,
        course_slug: str,
        question_id: str,
        batch_size: int = 15,
        use_rag: bool = True,
        background_tasks: BackgroundTasks = BackgroundTasks(),
        db: Session = Depends(db_manager.get_session),
        grader: GradingService = Depends(get_grading_service)
):
    """Step 4: Trigger the actual grading worker for a specific question."""
    from model.model import Grade
    ungraded = db.query(Grade.id).filter(
        Grade.assignment_id == assignment_id,
        Grade.question_id == question_id,
        Grade.confidence_score == 0.0
    ).all()

    grade_ids = [g[0] for g in ungraded]
    partitions = [grade_ids[i:i + batch_size] for i in range(0, len(grade_ids), batch_size)]

    for chunk in partitions:
        background_tasks.add_task(grader.grade_batch, db=db, grade_ids=chunk, course_slug=course_slug, use_rag=use_rag)

    return {"message": f"Dispatched {len(grade_ids)} grading jobs across {len(partitions)} batches."}