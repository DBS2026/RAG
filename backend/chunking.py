"""
backend/chunking.py

Converts page records (from pdf_processor) into chunks suitable for
embedding + retrieval.

Section-aware chunking: we detect common datasheet headings (Absolute
Maximum Ratings, Pin Description, Electrical Characteristics, ...) via
config.SECTION_HEADING_PATTERNS and split the page text at those
boundaries, instead of embedding an entire page as one chunk. Falls back
to size-based sliding-window splitting if no headings are found.

Tables get their own chunk per table (not flattened into page text), kept
as markdown for embedding + a JSON-serialized row list in metadata so the
UI can render an actual DataFrame instead of plain text.

Each chunk also gets lightweight, non-AI metadata tags:
  - category   (coarse classification via keyword counting)
  - keywords   (matches against a small controlled vocabulary)
  - manufacturer / component / revision (passed through from the page record)
"""

import json
import logging
import re
import uuid

import config
from backend import generator

logger = logging.getLogger(__name__)

_HEADING_ALTERNATION = "|".join(f"(?:{p})" for p in config.SECTION_HEADING_PATTERNS)
# Heading must appear at the start of a line (allowing a numbering prefix
# like "7.3") to avoid matching the phrase mid-sentence, e.g.
# "...refer to the Pin Description above."
_HEADING_LINE_PATTERN = re.compile(
    rf"^\s*(?:[\dA-Z]+(?:\.\d+)*\s+)?({_HEADING_ALTERNATION})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Simple sliding-window splitter for text that exceeds max_chars."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    pieces = []
    start = 0
    while start < len(text):
        end = start + max_chars
        piece = text[start:end]
        if piece.strip():
            pieces.append(piece)
        start = end - overlap  # overlap keeps context continuous across pieces
    return pieces


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Splits page text into (section_name, section_text) pairs using
    recognizable datasheet headings. Returns [] if no headings are found,
    signaling the caller to fall back to size-based splitting.
    """
    matches = list(_HEADING_LINE_PATTERN.finditer(text))
    if not matches:
        return []

    sections = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Overview", preamble))

    for idx, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((section_name, body))

    # Merge sections too short to be useful on their own into the previous one
    merged: list[list[str]] = []
    for name, body in sections:
        if merged and len(body) < config.MIN_SECTION_CHARS:
            merged[-1][1] += "\n\n" + name + "\n" + body
        else:
            merged.append([name, body])

    return [(name, body) for name, body in merged]


def _classify_category(text: str) -> str:
    """Coarse category via keyword counting — no LLM call needed."""
    lowered = text.lower()
    best_category, best_count = "General", 0
    for category, keywords in config.CATEGORY_KEYWORDS.items():
        count = sum(lowered.count(kw) for kw in keywords)
        if count > best_count:
            best_category, best_count = category, count
    return best_category


def _extract_keywords(text: str) -> list[str]:
    """Tags a chunk with any controlled-vocabulary terms it contains."""
    found = []
    for term in config.KEYWORD_VOCAB:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            found.append(term)
        if len(found) >= config.MAX_KEYWORDS_PER_CHUNK:
            break
    return found


def _base_chunk_fields(record: dict) -> dict:
    """Fields shared by every chunk produced from a given page record."""
    return {
        "document": record["document"],
        "page": record["page"],
        "figures": record["figures"],
        "tables_refs": record["tables_refs"],
        "image_path": record["image_path"],
        "figure_crops": record.get("figure_crops", {}),
        "figure_captions": record.get("figure_captions", {}),
        "manufacturer": record.get("manufacturer", "Unknown"),
        "component": record.get("component", "Unknown"),
        "revision": record.get("revision", "Unknown"),
    }


def build_chunks(page_records: list[dict]) -> list[dict]:
    """
    Takes page records and returns a flat list of chunk dicts:

    {
        "id": "uuid",
        "text": "section (or table markdown) text, used for embedding",
        "document": "TPS5430.pdf",
        "page": 17,
        "section": "Absolute Maximum Ratings",
        "category": "Power Supply",
        "keywords": ["VIN", "current", "thermal"],
        "manufacturer": "Texas Instruments",
        "component": "TPS5430",
        "revision": "E",
        "figures": ["Figure 6"],
        "figure_crops": {"Figure 6": "data/figures/TPS5430_p17_Figure_6.png"},
        "tables_refs": ["Table 4"],
        "table_rows": [[...], ...] or None,
        "image_path": "data/pages/TPS5430_p17.png"
    }
    """
    chunks = []

    for record in page_records:
        base = _base_chunk_fields(record)
        sections = _split_into_sections(record["text"])

        if not sections:
            pieces = _split_long_text(
                record["text"], config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP_CHARS
            )
            sections = [("Page Content", piece) for piece in pieces]

        for section_name, section_text in sections:
            for piece in _split_long_text(
                section_text, config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP_CHARS
            ):
                chunk_text = f"{section_name}\n{piece}"
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "section": section_name,
                    "category": _classify_category(chunk_text),
                    "keywords": _extract_keywords(chunk_text),
                    "table_rows": None,
                    **base,
                })

        # One chunk PER TABLE (not merged) so each keeps its own row data
        # for DataFrame rendering and isn't diluted by other tables' text.
        for table in record.get("tables", []):
            chunk_text = f"Table Data\n{table['markdown']}"
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "section": "Table Data",
                "category": _classify_category(chunk_text),
                "keywords": _extract_keywords(chunk_text),
                "table_rows": table["rows"],
                **base,
            })

        # One searchable chunk PER FIGURE, built from its printed caption plus a
        # vision-model summary of the cropped image. Indexing the caption
        # explicitly (not just relying on the vision summary to mention it)
        # means an exact caption phrase in the question always matches, while
        # the vision summary still lets an unlabeled question ("how is flash
        # connected?") find the figure via what it actually shows.
        if config.ENABLE_FIGURE_SUMMARIES:
            captions = record.get("figure_captions", {})
            for label, crop_path in record.get("figure_crops", {}).items():
                caption = captions.get(label, "")
                summary = None
                try:
                    summary = generator.summarize_figure(crop_path, label, caption)
                except Exception:
                    logger.exception("Figure summarization crashed for %s (%s)", label, crop_path)

                if not summary:
                    continue  # skip the chunk rather than index an empty/placeholder summary

                chunk_text_parts = [label]
                if caption:
                    chunk_text_parts.append(caption)
                chunk_text_parts.append(summary)
                chunk_text = "Figure Summary: " + "\n\n".join(chunk_text_parts)

                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "section": "Figure Summary",
                    "category": _classify_category(chunk_text),
                    "keywords": _extract_keywords(chunk_text),
                    "table_rows": None,
                    **base,
                    "figures": [label],
                })

    return chunks