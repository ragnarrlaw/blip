"""
shared Moodle Web Services API — talk HTTP with Moodle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from pydantic import Field

from moodle.error_hints import _HINTS

import requests

log = logging.getLogger("blip.moodle.api")


class AmbiguousMatch(Exception):
    """a given string is too close to more than one candidate to choose safely."""
    def __init__(self, *msg):
        super().__init__(*msg)
        self.msg = msg

    def __str__(self) -> str:
        return f"{str(self.msg)} - strings too close to more than one candidate to choose safely."


def flatten(params, prefix=""):
    """moodle wants PHP-array keys: courseids[0]=17, options[ids][0]=1."""
    out = {}
    items = params.items() if isinstance(params, dict) else enumerate(params)
    for k, v in items:
        key = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, (dict, list)):
            out.update(flatten(v, key))
        elif isinstance(v, bool):
            out[key] = int(v)
        else:
            out[key] = v
    return out


def get_token(base: str, user: str, password: str, service: str) -> str:
    """get the token (not the session token) from the moodle api - login/token.php"""
    r = requests.post(f"{base}/login/token.php",
                      data={"username": user, "password": password, "service": service}, timeout=30)
    log.debug("POST %s", r.url)
    r.raise_for_status()
    j = r.json()
    if "token" in j:
        log.debug("token minted (%s…)", j["token"][:6])
        return j["token"]
    code = j.get("errorcode", "")
    raise RuntimeError(f"token request failed [{code}]: {j.get('error', j)}\n"
                       f"  -> {_HINTS.get(code, 'Check the service shortname / web-service config.')}")


def call(base: str, token: str, fn: str, **params):
    data = flatten(params)
    data.update({"wstoken": token, "wsfunction": fn, "moodlewsrestformat": "json"})
    r = requests.post(f"{base}/webservice/rest/server.php", data=data, timeout=60)
    log.debug("POST %s (%s)", r.url, fn)
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and "exception" in j:
        code = j.get("errorcode", "")
        raise RuntimeError(f"{fn} failed [{code}]: {j.get('message', j)}\n"
                           f"  -> {_HINTS.get(code, 'See /admin/webservice/documentation.php.')}")
    return j


# ---- name resolution -------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def match_by_name(name: str, items: list[dict], key: str = "name",
                  floor: float = 0.6, margin: float = 0.1) -> dict:
    """
    exact (normalised) wins, else best fuzzy must clear `floor` and beat the runner-up by
    `margin`, so an ambiguous name fails loud instead of guessing (protects grades).
    """
    target = _norm(name)
    scored = sorted(
        ((1.0 if _norm(i[key]) == target else SequenceMatcher(None, target, _norm(i[key])).ratio(), i)
         for i in items),
        key=lambda x: -x[0])
    if not scored:
        raise LookupError(f"nothing to match {name!r}")
    best_score, best = scored[0]
    if best_score == 1.0:
        return best
    if best_score < floor:
        raise LookupError(f"no match for {name!r}; candidates: {[i[key] for _, i in scored[:5]]}")
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score - second < margin:
        raise AmbiguousMatch(f"{name!r} is ambiguous: {[(round(s, 2), i[key]) for s, i in scored[:3]]}")
    return best


# TODO: Convert this to a generic target - ResolvedTarget (Target anything on the LMS)
@dataclass(frozen=True)
class ResolvedQuiz:
    courseid: int = Field(description="course id")
    quizid: int = Field(description="quiz id")
    cmid: int = Field(description="course module id")
    report_url: str = Field(description="quiz report url with the responses table")


def report_url(base: str, cmid: int) -> str:
    return f"{base.rstrip('/')}/mod/quiz/report.php?id={cmid}&mode=responses"


def resolve_course_id(call_fn, course_shortname: str) -> int:
    courses = call_fn("core_course_get_courses_by_field",
                      field="shortname", value=course_shortname).get("courses", [])
    if not courses:
        raise LookupError(f"no course with shortname {course_shortname!r} (grader access?)")
    return courses[0]["id"]


def resolve_quiz(call_fn, base: str, courseid: int, quiz_name: str) -> ResolvedQuiz:
    quizzes = call_fn("mod_quiz_get_quizzes_by_courses", courseids=[courseid]).get("quizzes", [])
    q = match_by_name(quiz_name, quizzes)
    return ResolvedQuiz(courseid, q["id"], q["coursemodule"], report_url(base, q["coursemodule"]))