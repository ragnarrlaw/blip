import hashlib
import os
import time
from pathlib import Path
from fastapi import UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc

from config.conf import Config
from model.model import ContentItem, ContentItemVersion
from services.processing import process_and_embed_document


def calculate_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def ingest_course_file(
    db: Session,
    config: Config,
    course_id: int,
    item_key: str,
    filename: str,
    content_kind: str,
    file_bytes: bytes,
    background_tasks: BackgroundTasks,
) -> ContentItemVersion:
    """
    Entry point for all incoming files. Hashes content, manages versions,
    and triggers background processing if the content requires it.
    """
    file_hash = calculate_hash(file_bytes)

    # 1. Resolve or create the parent ContentItem
    item = db.query(ContentItem).filter(ContentItem.item_key == item_key).first()
    if not item:
        item = ContentItem(
            course_id=course_id,
            item_key=item_key,
            name=filename,
            content_kind=content_kind,
            last_updated=int(time.time()),
        )
        db.add(item)
        db.flush()

    # 2. Check for duplicate versions (skip identical bytes)
    existing_version = (
        db.query(ContentItemVersion)
        .filter(
            ContentItemVersion.content_item_id == item.id,
            ContentItemVersion.content_hash == file_hash,
        )
        .first()
    )

    if existing_version:
        # If it crashed previously, 'indexed_at' will be None.
        if not existing_version.indexed_at:
            log.info(f"Resuming failed vectorization for {filename}")
            if content_kind not in ["tutorial", "template"]:
                background_tasks.add_task(
                    process_and_embed_document,
                    version_id=existing_version.id,
                    config=config,
                )
        return existing_version

    # 3. Determine new version number
    last_version = (
        db.query(ContentItemVersion)
        .filter(ContentItemVersion.content_item_id == item.id)
        .order_by(desc(ContentItemVersion.version))
        .first()
    )

    new_version_num = (last_version.version + 1) if last_version else 1

    # 4. Save file to disk permanently
    course_slug = config.course_slug(
        str(course_id)
    )  # Usually course_code, simplified here
    save_dir = config.content_upload_dir(course_slug)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / f"v{new_version_num}_{filename}"
    file_path.write_bytes(file_bytes)

    # 5. Record the new version in the database
    new_version = ContentItemVersion(
        content_item_id=item.id,
        version=new_version_num,
        filename=filename,
        path=str(file_path),
        content_hash=file_hash,
        size_bytes=len(file_bytes),
        downloaded_at=str(int(time.time())),
    )
    db.add(new_version)
    db.commit()
    db.refresh(new_version)

    # 6. Content Router: Dispatch to processing queue based on kind
    # We skip "tutorial" and "template" as they have no contextual value for RAG
    if content_kind not in ["tutorial", "template"]:
        background_tasks.add_task(
            process_and_embed_document, version_id=new_version.id, config=config
        )

    return new_version

