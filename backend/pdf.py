"""
backend/pdf_processor.py

Turns a datasheet PDF into a list of "page records": text, tables (as
markdown + raw rows), a rendered full-page image, cropped figure images,
and any Figure/Table references found on that page.
"""

import os
import re
import logging

import pdfplumber
import fitz  # PyMuPDF

import config
from backend import generator

logger = logging.getLogger(__name__)

FIGURE_PATTERN = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
TABLE_PATTERN = re.compile(r"\bTable\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
_REVISION_RE = re.compile(config.REVISION_PATTERN, re.IGNORECASE)


def _find_references(text: str, pattern: re.Pattern, label: str) -> list[str]:
    seen = []
    for match in pattern.finditer(text or ""):
        ref = f"{label} {match.group(1)}"
        if ref not in seen:
            seen.append(ref)
    return seen


def _render_page_image(fitz_doc: "fitz.Document", page_number: int, doc_name: str) -> str:
    safe_doc_name = os.path.splitext(doc_name)[0].replace(" ", "_")
    out_path = os.path.join(config.PAGES_DIR, f"{safe_doc_name}_p{page_number}.png")

    if os.path.exists(out_path):
        return out_path

    page = fitz_doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    pix.save(out_path)
    return out_path


def _crop_figures(
    fitz_doc: "fitz.Document", page_number: int, doc_name: str, figure_labels: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    if not figure_labels:
        return {}, {}

    page = fitz_doc.load_page(page_number - 1)
    safe_doc_name = os.path.splitext(doc_name)[0].replace(" ", "_")
    crops = {}
    captions = {}

    for label in figure_labels:
        out_path = os.path.join(
            config.FIGURES_DIR,
            f"{safe_doc_name}_p{page_number}_{label.replace(' ', '_')}.png",
        )

        try:
            matches = page.search_for(label)
            if not matches:
                if os.path.exists(out_path):
                    crops[label] = out_path
                continue

            caption_rect = matches[0]

            # Grab the full caption line, not just the label itself — a full
            # width band at the label's vertical position, e.g. "Figure 8-2:
            # ESP32-C3 module reference schematic" instead of just "Figure 8-2".
            caption_band = fitz.Rect(
                page.rect.x0, caption_rect.y0 - 2, page.rect.x1, caption_rect.y1 + 2
            )
            caption_text = re.sub(r"\s+", " ", page.get_text("text", clip=caption_band) or "").strip()
            if caption_text:
                captions[label] = caption_text

            if os.path.exists(out_path):
                crops[label] = out_path
                continue

            crop_rect = fitz.Rect(
                page.rect.x0 + config.FIGURE_CROP_SIDE_MARGIN,
                max(page.rect.y0, caption_rect.y0 - config.FIGURE_CROP_BAND_HEIGHT),
                page.rect.x1 - config.FIGURE_CROP_SIDE_MARGIN,
                min(page.rect.y1, caption_rect.y1 + config.FIGURE_CROP_PADDING_BELOW),
            )
            if crop_rect.is_empty or crop_rect.height < 20:
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=crop_rect)
            pix.save(out_path)
            crops[label] = out_path
        except Exception:
            logger.exception("Failed to crop %s on page %s of %s", label, page_number, doc_name)
            continue

    return crops, captions


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Rebuilds markdown from a headers+body row list (rows[0] is the header)."""
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in body:
        padded = (row + [""] * len(header))[: len(header)]
        md_lines.append("| " + " | ".join(padded) + " |")
    return "\n".join(md_lines)


def _table_quality_is_low(rows: list[list[str]]) -> bool:
    """
    Flags tables likely mangled by merged cells: pdfplumber's line/text
    clustering strategies tend to leave a high fraction of cells blank
    when cells span multiple rows or columns in the source PDF.
    """
    total_cells = sum(len(row) for row in rows)
    if total_cells == 0:
        return True
    empty_cells = sum(1 for row in rows for cell in row if not str(cell).strip())
    return (empty_cells / total_cells) > config.TABLE_EMPTY_CELL_RATIO_THRESHOLD


def _crop_table_image(page: "fitz.Page", bbox: tuple[float, float, float, float]) -> bytes | None:
    """Renders the region of the page covered by a table's bounding box to PNG bytes."""
    try:
        crop_rect = fitz.Rect(*bbox)
        if crop_rect.is_empty:
            return None
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=crop_rect)
        return pix.tobytes("png")
    except Exception:
        logger.exception("Failed to crop table region for structured extraction.")
        return None


def _looks_like_continuation(page_text: str) -> bool:
    lowered = (page_text or "")[:400].lower()  # continuation notes appear near the top
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in config.TABLE_CONTINUATION_MARKERS)


def _normalize_header(header_row: list[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", " ", str(c).strip().lower()) for c in header_row)


def _merge_continued_table(prev_table: dict, next_table: dict) -> dict:
    """
    Merges a table that continues from the previous page into next_table:
    keeps next_table's header, appends prev_table's body rows before
    next_table's own body rows so reading order is preserved.
    """
    merged_rows = [next_table["rows"][0]] + prev_table["rows"][1:] + next_table["rows"][1:]
    return {
        "markdown": _rows_to_markdown(merged_rows),
        "rows": merged_rows,
        "spans_pages": True,
    }


def _table_to_markdown(table: list[list]) -> tuple[str, list[list[str]]]:
    rows = [[str(c).strip() if c is not None else "" for c in row] for row in table]
    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return "", []

    header, body = rows[0], rows[1:]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in body:
        padded = (row + [""] * len(header))[: len(header)]
        md_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(md_lines), rows


def _extract_document_metadata(first_pages_text: str, doc_name: str) -> dict:
    manufacturer = "Unknown"
    for name in config.KNOWN_MANUFACTURERS:
        if name.lower() in first_pages_text.lower():
            manufacturer = name
            break

    stem = os.path.splitext(doc_name)[0]
    stem = re.sub(r"(?i)[\s_\-]*(datasheet|manual|rev\w*|v\d+)\s*$", "", stem)
    component = re.sub(r"[\s_]+", " ", stem).strip().upper() or "Unknown"

    revision_match = _REVISION_RE.search(first_pages_text)
    revision = revision_match.group(1) if revision_match else "Unknown"

    return {"manufacturer": manufacturer, "component": component, "revision": revision}


def process_pdf(pdf_path: str, doc_name: str) -> list[dict]:
    page_records = []
    fitz_doc = fitz.open(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Shielding against NoneType returns on graphical/corrupt layout fronts
            first_pages_text = "\n".join(
                (p.extract_text() or "") for p in pdf.pages[:2]
            )
            doc_meta = _extract_document_metadata(first_pages_text, doc_name)
            prev_table = None  # last valid table on the previous page, for continuation merging

            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                try:
                    text = page.extract_text() or ""

                    # First Pass: Extract structural objects using geometric line intersections.
                    # find_tables (not extract_tables) also gives us each table's .bbox, which
                    # we need to crop the region for the structured-vision fallback below.
                    table_settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
                    table_objs = page.find_tables(table_settings=table_settings)

                    valid_objs = [t for t in table_objs if t.extract() and len(t.extract()) >= 2]

                    # Second Pass: If line tracking failed, fallback to text whitespace clustering
                    if not valid_objs:
                        table_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
                        table_objs = page.find_tables(table_settings=table_settings)
                    else:
                        table_objs = valid_objs

                    fitz_page = fitz_doc.load_page(page_number - 1)

                    tables = []
                    for t_obj in table_objs:
                        t = t_obj.extract()
                        markdown, rows = _table_to_markdown(t)

                        if not rows or len(rows) < 2:
                            continue

                        headers = rows[0]

                        # Verify Column Width Constraints
                        if len(headers) > 15 or len(headers) < 2:
                            continue

                        # Verify valid token content counts
                        valid_headers = sum(bool(str(h).strip()) for h in headers)
                        if valid_headers < 2:
                            continue

                        # Aggressive Header Threshold Drop
                        long_header = max(len(str(h)) for h in headers)
                        if long_header > 45:
                            continue

                        # Robust Contextual Group Rejection
                        joined = " ".join(headers).lower()
                        if (
                            ("figure" in joined and "schematics" in joined)
                            or ("block diagram" in joined)
                        ):
                            continue

                        table_entry = {"markdown": markdown, "rows": rows}

                        # Merged-cell / complex-layout fallback: if the geometric
                        # extraction left a suspicious fraction of cells blank,
                        # re-read the same region with the vision model instead.
                        if config.ENABLE_STRUCTURED_TABLE_FALLBACK and _table_quality_is_low(rows):
                            image_bytes = _crop_table_image(fitz_page, t_obj.bbox)
                            if image_bytes:
                                structured = generator.structure_table_from_image(image_bytes)
                                if structured and structured.get("headers") and structured.get("rows"):
                                    structured_rows = [structured["headers"]] + structured["rows"]
                                    table_entry = {
                                        "markdown": _rows_to_markdown(structured_rows),
                                        "rows": structured_rows,
                                    }

                        # Multi-page continuation: if this is the first table on the page
                        # and either its header matches the previous page's last table, or
                        # the page text opens with a "(continued)"-style marker, merge it
                        # into the running table instead of treating it as separate.
                        if (
                            prev_table is not None
                            and not tables
                            and (
                                _normalize_header(table_entry["rows"][0]) == _normalize_header(prev_table["rows"][0])
                                or _looks_like_continuation(text)
                            )
                        ):
                            table_entry = _merge_continued_table(prev_table, table_entry)

                        tables.append(table_entry)

                    prev_table = tables[-1] if tables else None

                    figures = _find_references(text, FIGURE_PATTERN, "Figure")
                    table_refs = _find_references(text, TABLE_PATTERN, "Table")

                    image_path = _render_page_image(fitz_doc, page_number, doc_name)
                    figure_crops, figure_captions = _crop_figures(fitz_doc, page_number, doc_name, figures)

                    if not text.strip() and not tables:
                        continue

                    page_records.append({
                        "document": doc_name,
                        "page": page_number,
                        "text": text,
                        "tables": tables,
                        "figures": figures,
                        "figure_crops": figure_crops,
                        "figure_captions": figure_captions,
                        "tables_refs": table_refs,
                        "image_path": image_path,
                        **doc_meta,
                    })
                except Exception:
                    logger.exception("Failed to process page %s of %s — skipping page.",
                                      page_number, doc_name)
                    continue
    finally:
        fitz_doc.close()

    return page_records