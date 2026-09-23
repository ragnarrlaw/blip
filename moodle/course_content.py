"""
Moodle course-content fetch over Web Services (no browser). Flattens
core_course_get_contents into ContentItems, builds token'd download URLs, hashes bytes.

Text/parse extraction now lives in `parsing.document` (Docling) — this module is purely the
Moodle-side fetch/flatten/download, so the two concerns don't tangle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ContentItem:
    item_key: str
    course_id: int
    cmid: int
    module_name: str
    filename: str
    fileurl: str
    timemodified: int
    mimetype: str = ""
    filesize: int = 0


def flatten_contents(course_id: int, sections: list[dict]) -> list[ContentItem]:
    items: list[ContentItem] = []
    for sec in sections:
        for mod in sec.get("modules", []):
            cmid, mod_name = mod.get("id", 0), mod.get("name", "")
            for c in mod.get("contents", []):
                if c.get("type") != "file":
                    continue
                filepath, filename = c.get("filepath", "/"), c.get("filename", "")
                items.append(ContentItem(
                    item_key=f"{cmid}:{filepath}{filename}",
                    course_id=course_id, cmid=cmid, module_name=mod_name,
                    filename=filename, fileurl=c.get("fileurl", ""),
                    timemodified=c.get("timemodified", 0),
                    mimetype=c.get("mimetype", ""), filesize=c.get("filesize", 0)))
    return items


def pluginfile_url(fileurl: str, token: str) -> str:
    sep = "&" if "?" in fileurl else "?"
    return f"{fileurl}{sep}token={token}"


def download_bytes(fileurl: str, token: str) -> bytes:
    import requests
    r = requests.get(pluginfile_url(fileurl, token), timeout=120)
    r.raise_for_status()
    return r.content


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()