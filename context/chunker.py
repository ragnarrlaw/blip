"""
Recursive character chunker.

Splits on the coarsest separator present ("\\n\\n", then "\\n", then " ", then characters), and
any piece still longer than chunk_size is split again with the next separator -- so no chunk
exceeds chunk_size. Neighbouring pieces are merged up to chunk_size, and consecutive chunks share
up to chunk_overlap characters.

chunk_size and chunk_overlap count CHARACTERS, not tokens. Keep chunk_size under the embedding
model's window (all-MiniLM-L6-v2: 256 tokens, roughly 1000 characters of English), or the tail of
every chunk is truncated before embedding and retrieval never sees it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_SEPARATORS = ("\n\n", "\n", " ", "")


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    heading: str = ""


def _merge(pieces: list[str], sep: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Join small pieces into chunks of at most chunk_size, carrying an overlap tail forward."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    sep_len = len(sep)
    for piece in pieces:
        added = len(piece) + (sep_len if current else 0)
        if current and length + added > chunk_size:
            chunks.append(sep.join(current))
            # Drop from the front until what is left is a valid overlap AND the next piece fits.
            while current and (length > chunk_overlap or length + len(piece) + sep_len > chunk_size):
                removed = current.pop(0)
                length -= len(removed) + (sep_len if current else 0)
            added = len(piece) + (sep_len if current else 0)
        current.append(piece)
        length += added
    if current:
        chunks.append(sep.join(current))
    return chunks


def _split(text: str, separators: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    sep, rest = separators[-1], []
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            sep, rest = candidate, separators[i + 1:]
            break
    pieces = text.split(sep) if sep else list(text)

    out: list[str] = []
    small: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            small.append(piece)
            continue
        if small:
            out.extend(_merge(small, sep, chunk_size, chunk_overlap))
            small = []
        if rest:
            out.extend(_split(piece, rest, chunk_size, chunk_overlap))
        else:  # no finer separator left: hard cut
            out.extend(piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size))
    if small:
        out.extend(_merge(small, sep, chunk_size, chunk_overlap))
    return out


def recursive_character_split(text: str, chunk_size: int = 1500, chunk_overlap: int = 150,
                              separators: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    seps = list(separators or DEFAULT_SEPARATORS)
    return [c for c in _split(text, seps, chunk_size, chunk_overlap) if c.strip()]


_HEADING = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def generate_chunks(text: str, chunk_size: int = 1500, overlap: int = 150,
                    separators: list[str] | tuple[str, ...] | None = None) -> list[Chunk]:
    """Chunks with the most recent markdown heading carried forward as context."""
    chunks: list[Chunk] = []
    heading = ""
    for i, piece in enumerate(recursive_character_split(text, chunk_size, overlap, separators)):
        # The heading in force at the START of the chunk is the one it belongs under.
        chunk_heading = heading
        found = _HEADING.findall(piece)
        if found and piece.lstrip().startswith("#"):
            chunk_heading = found[0][1].strip()
        if found:
            heading = found[-1][1].strip()
        chunks.append(Chunk(text=piece.strip(), index=i, heading=chunk_heading))
    return chunks
