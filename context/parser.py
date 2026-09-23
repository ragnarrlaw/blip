from __future__ import annotations

import logging
import os
import re
import tempfile

from context.chunker import Chunk, generate_chunks

log = logging.getLogger("blip.parsing")

# Docling options. Change via configure_docling() BEFORE the first parse().
DO_OCR = True               # needed for scanned / image-only PDFs; slower on digital ones
DO_TABLE_STRUCTURE = True
NUM_THREADS = None          # None -> all CPU cores

_converter = None


class DoclingNotInstalled(RuntimeError):
    """Docling is a required dependency; raised if it's missing."""


def configure_docling(*, do_ocr: bool | None = None, do_table_structure: bool | None = None,
                      num_threads: int | None = None) -> None:
    """set converter options before the first parse(); forces a rebuild on next use."""
    global DO_OCR, DO_TABLE_STRUCTURE, NUM_THREADS, _converter
    if do_ocr is not None:
        DO_OCR = do_ocr
    if do_table_structure is not None:
        DO_TABLE_STRUCTURE = do_table_structure
    if num_threads is not None:
        NUM_THREADS = num_threads
    _converter = None


def _get_file_extension(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _get_converter():
    global _converter
    if _converter is None:
        import multiprocessing
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            try:
                from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            except ImportError:
                from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
        except ImportError as e:
            raise DoclingNotInstalled("docling is required — pip install docling transformers") from e

        opts = PdfPipelineOptions(do_ocr=DO_OCR, do_table_structure=DO_TABLE_STRUCTURE)
        opts.accelerator_options = AcceleratorOptions(
            num_threads=NUM_THREADS or multiprocessing.cpu_count(), device=AcceleratorDevice.AUTO)

        # PDF uses the options above; DOCX, PPTX, XLSX, HTML, MD keep Docling's defaults.
        # (Previously DocumentConverter() was built without these options, so configure_docling()
        # had no effect.)
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        log.info("docling converter ready (ocr=%s, tables=%s)", DO_OCR, DO_TABLE_STRUCTURE)
    return _converter


def sanitize_text(text: str) -> str:
    """Strips PII and specific boilerplate before chunking."""
    if not text:
        return ""

    # 1. Strip all email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # 2. Strip phone numbers (adjust regex for Sri Lankan / international formats if needed)
    text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{2,3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)

    # 3. Strip specific university tags or boilerplate phrases
    boilerplate = [
        r"University of Colombo School of Computing",
        r"UCSC",
        r"All rights reserved",
        r"Course Coordinator:.*?\n"
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean up any leftover double spaces or empty lines created by the removal
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class ParsedDoc:
    def __init__(self, document):
        self._doc = document

    @property
    def text(self) -> str:
        # Docling natively exports to markdown, standardizing the format for the chunker
        return self._doc.export_to_markdown()

    def chunks(self, max_tokens: int = 1500, overlap: int = 150) -> list[Chunk]:
        # NB: despite the name, max_tokens is a character count (see chunker.py).
        return generate_chunks(sanitize_text(self.text), chunk_size=max_tokens, overlap=overlap)


def parse(filename: str, data: bytes) -> ParsedDoc:
    ext = _get_file_extension(filename)
    conv = _get_converter()
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        log.debug("parsing %s via %s", filename, path)
        return ParsedDoc(conv.convert(path).document)
    except Exception as e:
        log.error(f"Docling parse failed for {filename}: {e}")
        raise
    finally:
        os.unlink(path)


def parse_to_markdown(filename: str, data: bytes) -> str:
    """The unit the indexer stores: parse once, keep the markdown, re-chunk it as often as needed."""
    return parse(filename, data).text


def extract_text(filename: str, data: bytes) -> str:
    return parse_to_markdown(filename, data)
