import time
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db.db import db_manager
from model.model import ContentItemVersion, ContentItem, Course
from context.parser import parse, sanitize_text
from context.retriever import ChromaRetriever

log = logging.getLogger("blip.processing")


def process_and_embed_document(version_id: int, config):
    """
    Background worker that extracts text, chunks it, embeds it into ChromaDB,
    and evicts previous versions of the same document from the active index.
    """
    # Create an isolated database session for the background worker
    db: Session = next(db_manager.get_session())
    retriever = ChromaRetriever(config)

    try:
        # Fetch relationships
        version = (
            db.query(ContentItemVersion)
            .filter(ContentItemVersion.id == version_id)
            .first()
        )
        if not version:
            return

        item = version.content_item
        course = db.query(Course).filter(Course.id == item.course_id).first()
        course_slug = config.course_slug(course.course_code)
        collection_name = config.chroma_collection(course.course_code)

        # 1. Parse using Docling
        log.info(f"Parsing document {version.filename} (Version {version.version})")
        file_bytes = Path(version.path).read_bytes()
        parsed_doc = parse(version.filename, file_bytes)

        # Save parsed markdown to disk to skip parsing on future regrades
        parsed_dir = Path(config.blip_output_dir) / course_slug / "parsed_content"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        parsed_path = parsed_dir / f"{version.content_hash}.md"
        parsed_path.write_text(parsed_doc.text, encoding="utf-8")
        print(f"\n---> ATTENTION: Just saved markdown to: {parsed_path.absolute()} <--- \n")

        version.parsed_path = str(parsed_path)
        version.parsed_at = str(int(time.time()))
        db.commit()

        # 2. Chunk the document
        chunks = parsed_doc.chunks(
            max_tokens=config.blip_embed_max_tokens, overlap=config.blip_chunk_overlap
        )
        chunk_ids = [f"{item.item_key}_v{version.version}#{c.index}" for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "item_key": item.item_key,
                "version": version.version,
                "version_id": version.id,
                "filename": version.filename,
                "heading": c.heading,
            }
            for c in chunks
        ]

        # 3. Vector Eviction: Remove previous version's chunks from ChromaDB
        previous_versions = (
            db.query(ContentItemVersion)
            .filter(
                ContentItemVersion.content_item_id == item.id,
                ContentItemVersion.id != version.id,
                ContentItemVersion.evicted_at == None,
            )
            .all()
        )

        for prev_v in previous_versions:
            if prev_v.chunk_ids:
                log.info(f"Evicting v{prev_v.version} vectors for {item.item_key}")
                retriever.delete(course_slug, collection_name, prev_v.chunk_ids)
            prev_v.evicted_at = str(int(time.time()))

        # 4. Ingest new chunks into ChromaDB
        retriever.index(
            course_name=course_slug,
            collection_name=collection_name,
            chunk_ids=chunk_ids,
            texts=texts,
            metadatas=metadatas,
        )

        # 5. Finalize Database State
        version.chunk_ids = chunk_ids
        version.indexed_at = str(int(time.time()))
        db.commit()
        log.info(
            f"Successfully embedded {len(chunk_ids)} chunks for {version.filename}"
        )

    except Exception as e:
        log.error(f"Processing failed for version_id {version_id}: {str(e)}")
        version.parse_error = str(e)
        db.commit()
    finally:
        db.close()

