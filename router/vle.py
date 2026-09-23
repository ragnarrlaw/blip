from typing import Any, Dict, List, Union
from fastapi import APIRouter, Depends, HTTPException, Query

from moodle.dependencies import get_moodle_session
from moodle.moodle_session import MoodleSession
from services import vle_discovery

router = APIRouter(prefix="/vle", tags=["VLE Discovery"])


@router.get("/courses", response_model=List[Dict[str, Any]])
def list_lecturer_courses(
        moodle: MoodleSession = Depends(get_moodle_session)
):
    """Returns all courses the authenticated lecturer is enrolled in."""
    try:
        return vle_discovery.get_lecturer_courses(moodle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch courses: {str(e)}")


@router.get("/courses/{course_identifier}", response_model=Dict[str, Any])
def get_course_details(
        course_identifier: str,
        moodle: MoodleSession = Depends(get_moodle_session)
):
    """Returns metadata for a specific course (accepts course ID or shortname)."""
    try:
        return vle_discovery.get_course_overview(moodle, course_identifier)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch course details: {str(e)}")


@router.get("/courses/{course_identifier}/contents", response_model=Dict[str, Any])
def preview_course_contents(
        course_identifier: str,
        omit_keywords: List[str] = Query(default=["tutorial", "tute", "template"]),
        moodle: MoodleSession = Depends(get_moodle_session)
):
    try:
        return vle_discovery.get_raw_course_contents(moodle, course_identifier, omit_keywords)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/courses/{course_identifier}/assignments", response_model=List[Dict[str, Any]])
def list_course_assignments(
        course_identifier: str,
        moodle: MoodleSession = Depends(get_moodle_session)
):
    """Lists all quizzes/assignments configured in the course on Moodle."""
    try:
        return vle_discovery.get_course_assignments(moodle, course_identifier)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch assignments: {str(e)}")
