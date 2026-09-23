import logging
from typing import Any, Dict, List, Union, Tuple, Literal

from moodle.course_content import flatten_contents
from moodle.moodle_session import MoodleSession

log = logging.getLogger("blip.vle_discovery")

def _resolve_course_id(moodle: MoodleSession, course_identifier: Union[str, int]) -> int:
    """Accepts either an integer Moodle course ID or a shortname string."""
    if isinstance(course_identifier, int) or str(course_identifier).isdigit():
        return int(course_identifier)
    return moodle.resolve_course_id(str(course_identifier))


def get_lecturer_courses(moodle: MoodleSession) -> List[Dict[str, Any]]:
    """
    Fetches the authenticated lecturer's profile and returns all enrolled courses.
    """
    site_info = moodle.call("core_webservice_get_site_info")
    user_id = site_info.get("userid")
    raw_courses = moodle.call("core_enrol_get_users_courses", userid=user_id)

    return [
        {
            "id": c.get("id"),
            "shortname": c.get("shortname"),
            "fullname": c.get("fullname"),
            "displayname": c.get("displayname") or c.get("fullname"),
            "enrolled_students": c.get("enrolledusercount", 0),
            "summary": c.get("summary", ""),
        }
        for c in raw_courses
    ]


def get_course_overview(moodle: MoodleSession, course_identifier: Union[str, int]) -> Dict[str, Any]:
    """
    Fetches official metadata for a single course.
    """
    course_id = _resolve_course_id(moodle, course_identifier)
    res = moodle.call("core_course_get_courses_by_field", field="id", value=course_id)
    courses = res.get("courses", [])
    if not courses:
        raise LookupError(f"Course '{course_identifier}' not found on Moodle.")

    c = courses[0]
    return {
        "id": c.get("id"),
        "shortname": c.get("shortname"),
        "fullname": c.get("fullname"),
        "displayname": c.get("displayname", c.get("fullname")),
        "summary": c.get("summary", ""),
        "startdate": c.get("startdate"),
        "enddate": c.get("enddate"),
    }


def get_raw_course_contents(moodle: MoodleSession, course_identifier: Union[str, int],
                            omit_keywords: List[str] | None = None,
                            allowed_extensions: Tuple[str] = (".pdf", ".docx", ".txt", ".md", "pptx")) -> Dict[str, Any]:
    """
    Inspects course content on Moodle and classifies each file before indexing.
    """
    if omit_keywords is None:
        omit_keywords = ["tutorial", "tute", "template"]
    course_id = _resolve_course_id(moodle, course_identifier)
    sections = moodle.call("core_course_get_contents", courseid=course_id)
    flattened_items = flatten_contents(course_id, sections)

    categorized_items = []
    total_materials = 0
    total_omitted = 0

    for item in flattened_items:
        filename_lower = item.filename.lower()
        is_omitted = any(keyword in filename_lower for keyword in omit_keywords)
        is_supported = filename_lower.endswith(allowed_extensions)

        if not is_supported:
            category = "unsupported_format"
        elif is_omitted:
            category = "omitted_non_material"
            total_omitted += 1
        else:
            category = "indexable_material"
            total_materials += 1

        categorized_items.append({
            "item_key": item.item_key,
            "cmid": item.cmid,
            "module_name": item.module_name,
            "filename": item.filename,
            "fileurl": item.fileurl,
            "filesize": item.filesize,
            "timemodified": item.timemodified,
            "status": category
        })

    return {
        "course_id": course_id,
        "total_files": len(flattened_items),
        "indexable_materials": total_materials,
        "omitted_materials": total_omitted,
        "files": categorized_items,
    }


def get_course_assignments(moodle: MoodleSession, course_identifier: Union[str, int]) -> List[Dict[str, Any]]:
    """
    Lists quizzes and assessments available for automated grading.
    """
    course_id = _resolve_course_id(moodle, course_identifier)
    quizzes_res = moodle.call("mod_quiz_get_quizzes_by_courses", courseids=[course_id])
    quizzes = quizzes_res.get("quizzes", [])

    return [
        {
            "id": q.get("id"),
            "cmid": q.get("coursemodule"),
            "name": q.get("name"),
            "timeopen": q.get("timeopen"),
            "timeclose": q.get("timeclose"),
            "gradepass": q.get("gradepass"),
            "report_url": f"{moodle.base}/mod/quiz/report.php?id={q.get('coursemodule')}&mode=responses",
        }
        for q in quizzes
    ]
