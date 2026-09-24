import json
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from db.db import db_manager
from config.conf import get_config, Config
from services.master_question_extractor import generate_master_questions_from_file
from services.assignment_service import ingest_moodle_submissions, ingest_research_dataset, get_or_create_assignment
from services.report_generator import generate_assignment_report

router = APIRouter(prefix="/assignments", tags=["Assignments & Setup"])


@router.post("/master-questions/upload")
async def upload_master_questions(
        course_id: int = Form(...),
        cmid: int = Form(...),
        assignment_name: str = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(db_manager.get_session),
        config: Config = Depends(get_config),
):
    # Lock Master Questions to the specific Assignment
    assignment = get_or_create_assignment(db, course_id, cmid, assignment_name)

    temp_dir = Path(config.blip_output_dir) / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / file.filename

    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        count = generate_master_questions_from_file(db, config, assignment.id, temp_file)
        return {"status": "success", "assignment_id": assignment.id, "master_questions_created": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_file.unlink(missing_ok=True)


@router.post("/submissions/moodle")
async def upload_moodle_json(
        course_id: int = Form(...),
        cmid: int = Form(...),
        assignment_name: str = Form(...),
        file: UploadFile = File(...),
        db: Session = Depends(db_manager.get_session),
        config: Config = Depends(get_config),
):
    try:
        content = await file.read()
        raw_json = json.loads(content.decode("utf-8"))
        result = ingest_moodle_submissions(db, config, course_id, cmid, assignment_name, raw_json)
        return {"status": "success", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submissions/dataset")
def ingest_from_dataset_file(
        course_id: int,
        cmid: int,
        assignment_name: str,
        relative_dataset_path: str,
        db: Session = Depends(db_manager.get_session)
):
    """Bypasses LLM segmentation and ingests your cleaned research JSON directly."""
    path = Path(relative_dataset_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        result = ingest_research_dataset(db, course_id, cmid, assignment_name, raw_json)
        return {"status": "success", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{assignment_id}")
def export_grading_report(assignment_id: int, db: Session = Depends(db_manager.get_session)):
    """Generates and downloads the multi-sheet Excel report."""
    try:
        excel_file, name = generate_assignment_report(db, assignment_id)
        filename = f"{name.replace(' ', '_')}_Grading_Report.xlsx"

        return StreamingResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except ValueError as e:
        print("exception message @ export_grading_report: ", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print("exception message @ export_grading_report: ", e)
        raise HTTPException(status_code=500, detail=str(e))