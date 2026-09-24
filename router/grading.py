from collections import defaultdict
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from db.db import db_manager
from config.conf import get_config, Config
from model.model import Grade
from schema.schema import GradingExecutionFilter
from services.grading_service import GradingService

router = APIRouter(prefix="/grading", tags=["Grading Engine"])


def get_grading_service(cfg: Config = Depends(get_config)) -> GradingService:
    return GradingService(cfg)


@router.post("/execute/{assignment_id}")
def execute_flexible_grading(
    assignment_id: int,
    course_slug: str,
    filters: GradingExecutionFilter,
    background_tasks: BackgroundTasks,
    db: Session = Depends(db_manager.get_session),
    grader: GradingService = Depends(get_grading_service),
):
    """
    Dynamically filters the Grade table and dispatches batches to the grading engine.
    """
    query = db.query(Grade).filter(Grade.assignment_id == assignment_id)

    if filters.student_ids:
        query = query.filter(Grade.student_id.in_(filters.student_ids))
    if filters.question_ids:
        query = query.filter(Grade.question_id.in_(filters.question_ids))
    if filters.question_types:
        query = query.filter(Grade.question_type.in_(filters.question_types))

    if not filters.force_regrade:
        query = query.filter(Grade.confidence_score == 0.0)

    target_grades = query.all()
    if not target_grades:
        raise HTTPException(
            status_code=404,
            detail="No matching submissions found for the provided filters.",
        )

    # Group by question_id so each batch shares the same rubric and context
    grouped_jobs = defaultdict(list)
    for g in target_grades:
        grouped_jobs[g.question_id].append(g.id)

    total_batches = 0
    for q_id, grade_ids in grouped_jobs.items():
        partitions = [
            grade_ids[i : i + filters.batch_size]
            for i in range(0, len(grade_ids), filters.batch_size)
        ]
        for chunk in partitions:
            background_tasks.add_task(
                grader.grade_batch,
                db=db,
                grade_ids=chunk,
                course_slug=course_slug,
                use_rag=filters.use_rag,
                model_override=filters.model_name,
            )
            total_batches += 1

    return {
        "status": "execution_queued",
        "total_grades_targeted": len(target_grades),
        "questions_targeted": list(grouped_jobs.keys()),
        "batches_dispatched": total_batches,
        "configuration": {
            "use_rag": filters.use_rag,
            "model": filters.model_name or grader.provider.default_model,
            "force_regrade": filters.force_regrade,
        },
    }


@router.get("/results/{assignment_id}")
def get_grading_results(
    assignment_id: int, db: Session = Depends(db_manager.get_session)
):
    """Returns graded responses with awarded marks, confidence scores, and telemetry."""
    grades = (
        db.query(Grade)
        .filter(Grade.assignment_id == assignment_id, Grade.confidence_score > 0.0)
        .all()
    )

    return [
        {
            "id": g.id,
            "student_id": g.student_id,
            "question_id": g.question_id,
            "mark_assigned": g.mark_assigned,
            "max_mark": g.max_mark,
            "confidence_score": g.confidence_score,
            "justification": g.justifications,
            "params_used": g.params_used,
        }
        for g in grades
    ]

