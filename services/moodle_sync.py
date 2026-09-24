import logging
from typing import Dict, Any, List
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from config.conf import Config
from moodle.moodle_session import MoodleSession
from moodle.course_content import flatten_contents, download_bytes
from services.ingestion import ingest_course_file
from services.vle_discovery import get_course_overview
from model.model import Course

log = logging.getLogger("blip.sync")


def sync_course_materials(
    db: Session,
    config: Config,
    moodle: MoodleSession,
    course_shortname: str,
    omit_keywords: List[str],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    try:
        # 1. Resolve metadata and ensure Course exists in database
        course_meta = get_course_overview(moodle, course_shortname)
        course_id = course_meta["id"]

        course = db.query(Course).filter(Course.id == course_id).first()
        if not course:
            course = Course(
                id=course_id,
                course_code=course_meta["shortname"],
                course_name=course_meta["fullname"],
                enrollments=0,
            )
            db.add(course)
        else:
            course.course_code = course_meta["shortname"]
            course.course_name = course_meta["fullname"]
        db.commit()

        # 2. Fetch course structure
        sections = moodle.call("core_course_get_contents", courseid=course_id)
        moodle_items = flatten_contents(course_id, sections)

        results = {"total_found": len(moodle_items), "ingested": 0, "skipped": 0}

        # 3. Filter and dispatch ingestion
        for item in moodle_items:
            filename_lower = item.filename.lower()
            if not filename_lower.endswith((".pdf", ".md", ".txt", ".docx", ".pptx")):
                results["skipped"] += 1
                continue

            if any(word.lower() in filename_lower for word in omit_keywords):
                results["skipped"] += 1
                continue

            file_bytes = download_bytes(item.fileurl, moodle.token)

            ingest_course_file(
                db=db,
                config=config,
                course_id=course_id,
                item_key=item.item_key,
                filename=item.filename,
                content_kind="material",
                file_bytes=file_bytes,
                background_tasks=background_tasks,
            )
            results["ingested"] += 1

        return results

    except LookupError as e:
        raise ValueError(f"Course not found: {str(e)}")

