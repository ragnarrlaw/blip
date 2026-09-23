from typing import List

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Query
from sqlalchemy.orm import Session

from db.db import db_manager
from config.conf import get_config, Config
from moodle.dependencies import get_moodle_session
from moodle.moodle_session import MoodleSession
from services.moodle_sync import sync_course_materials

router = APIRouter(prefix="/sync", tags=["Moodle Sync"])

@router.post("/course/{course_shortname}")
def trigger_course_sync(
    course_shortname: str,
    background_tasks: BackgroundTasks,
    omit_keywords: List[str] = Query(default=["tutorial", "tute", "template"]),
    db: Session = Depends(db_manager.get_session),
    config: Config = Depends(get_config),
    moodle: MoodleSession = Depends(get_moodle_session)
):
    try:
        stats = sync_course_materials(
            db=db,
            config=config,
            moodle=moodle,
            course_shortname=course_shortname,
            omit_keywords=omit_keywords,
            background_tasks=background_tasks
        )
        return {
            "message": f"Sync completed for {course_shortname}",
            "stats": stats
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))