"""
backend/generator.py

Builds the prompt from retrieved chunks and calls Gemini to produce a
structured, citation-grounded answer.
"""

import hashlib
import json
import logging
import re
import sqlite3
from collections import OrderedDict

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger(__name__)

config.require_gemini_key()
_genai_client = genai.Client(api_key=config.GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """You are a datasheet engineering assistant used by students and \
engineers to understand electronic component datasheets.

Rules you must follow:
1. Answer ONLY using the provided CONTEXT below. Do not use outside knowledge.
2. If the context does not contain the answer, say so clearly — do not guess.
3. Every factual claim must include a citation in the form [Document, Page].
4. Structure your answer using these sections, and OMIT any section that
   doesn't apply to the question:

   **Summary**
   **Power Requirements** (voltage, current, if relevant)
   **Pin / Connection Details**
   **Required Components**
   **Warnings / Notes**
   **Datasheet References** (list of [Document, Page] citations used)

Keep it concise and technical — this is for engineers, not beginners.
Do NOT add your own confidence rating — that is computed separately.
"""

_answer_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(question: str, chunks: list[dict]) -> str:
    chunk_signature = "|".join(
        f"{c['metadata']['document']}:{c['metadata']['page']}:{c['metadata'].get('section', '')}"
        for c in chunks
    )
    raw = f"{question}::{chunk_signature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    if key in _answer_cache:
        _answer_cache.move_to_end(key)
        return _answer_cache[key]
    return None


def _cache_set(key: str, value: str):
    _answer_cache[key] = value
    _answer_cache.move_to_end(key)
    if len(_answer_cache) > config.ANSWER_CACHE_MAX_ITEMS:
        _answer_cache.popitem(last=False)


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    total_chars = 0

    for chunk in chunks:
        meta = chunk["metadata"]
        header = f"[Source: {meta['document']}, Page {meta['page']}"
        if meta.get("component") and meta["component"] != "Unknown":
            header += f", Component: {meta['component']}"
        if meta.get("section"):
            header += f", Section: {meta['section']}"
        header += "]"
        if meta.get("figures"):
            header += f" (mentions: {meta['figures']})"

        block = f"{header}\n{chunk['text']}"
        if total_chars + len(block) > config.MAX_CONTEXT_CHARS and blocks:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(blocks)


def _estimate_confidence(chunks: list[dict]) -> tuple[str, str]:
    """
    Computes a realistic, normalized confidence metric scaled between 0 and 1.0, 
    matching against the math boundaries of the updated configuration.
    """
    if not chunks:
        return "Low", "No matching content was found in the indexed datasheets."

    scores = [c["score"] for c in chunks if c.get("score") is not None]
    if not scores:
        return "High", "Direct structural lookup target found."

    # Baseline calculations
    base_rrf_max = (config.VECTOR_WEIGHT / (config.RRF_K + 1)) + (config.BM25_WEIGHT / (config.RRF_K + 1))
    
    # Scale calculation against safety max ceilings
    theoretical_max = base_rrf_max * config.MAX_SECTION_MULTIPLIER
    
    top_score = scores[0]
    normalized_score = top_score / theoretical_max

    if normalized_score >= 0.75:
        level = "High"
        reason = f"Highly correlated search matches identified across your indexes (Normalized match: {normalized_score:.1%})."
    elif normalized_score >= 0.40:
        level = "Medium"
        reason = f"Moderate semantic/structural intersections located. Cross-verify details manually (Normalized match: {normalized_score:.1%})."
    else:
        level = "Low"
        reason = f"Fragmented or single-term semantic matches. Content might be speculative (Normalized match: {normalized_score:.1%})."

    return level, reason


@retry(
    stop=stop_after_attempt(config.API_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.API_BACKOFF_BASE_SECONDS, min=1, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini(prompt: str) -> str:
    response = _genai_client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt,
    )
    return response.text


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return ("I couldn't find relevant information in the uploaded datasheets "
                "for this question. Try rephrasing, or upload the relevant datasheet.\n\n"
                "**Confidence:** Low — no matching content was found in the indexed datasheets.")

    cache_key = _cache_key(question, chunks)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    context = _format_context(chunks)
    prompt = f"{SYSTEM_INSTRUCTION}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"

    answer_text = _call_gemini(prompt)

    level, reason = _estimate_confidence(chunks)
    answer_text = f"{answer_text}\n\n**Confidence:** {level} — {reason}"

    _cache_set(cache_key, answer_text)
    return answer_text


_DIRECT_REF_PATTERN = re.compile(
    r"\b(?:Figure|Fig\.?|Table)\s+\d+[A-Za-z]?\b", re.IGNORECASE
)


def extract_direct_reference(question: str) -> str | None:
    match = _DIRECT_REF_PATTERN.search(question)
    return match.group(0) if match else None


# =====================================================================
# Query decomposition
#
# Long engineering questions ("Describe GPIOs, operating voltage, RAM,
# WiFi, package, boot mode...") lose recall when embedded and searched
# as a single vector, because the embedding becomes an unfocused blend
# of many topics. We ask the model to split such questions into a small
# set of focused sub-questions, each retrieved independently, then
# merge and dedupe the results before answering.
# =====================================================================

_DECOMPOSITION_INSTRUCTION = """You are a query planner for a datasheet search engine.

Given a user's question about an electronic component datasheet, decide whether it
asks about multiple distinct topics (e.g. several different pins, specs, or features)
or is already a single focused question.

- If it is a single focused question, return a JSON array with just that one question.
- If it covers multiple topics, split it into up to {max_subqueries} focused
  sub-questions, each one self-contained (repeat the component/subject if needed)
  and specific enough to search a datasheet with (e.g. "What is the operating
  voltage range?" rather than just "voltage").
- Do not answer the question. Do not add commentary.
- Respond with ONLY a JSON array of strings, nothing else.

Question: {question}
"""


def _heuristically_needs_decomposition(question: str) -> bool:
    """Cheap pre-filter so we don't spend an LLM call on simple questions."""
    if len(question) < config.DECOMPOSITION_MIN_CHARS:
        return False
    # Signals of a multi-part question: several commas/"and"s, or a list-like
    # structure separating distinct technical terms.
    separators = len(re.findall(r",| and | & |;", question, re.IGNORECASE))
    return separators >= 2


@retry(
    stop=stop_after_attempt(config.API_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.API_BACKOFF_BASE_SECONDS, min=1, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini_decomposition(prompt: str) -> str:
    response = _genai_client.models.generate_content(
        model=config.DECOMPOSITION_MODEL,
        contents=prompt,
    )
    return response.text


def decompose_question(question: str) -> list[str]:
    """
    Splits a multi-topic question into focused sub-questions.

    Returns a list containing just the original question if decomposition
    is disabled, unnecessary, or fails for any reason — callers can always
    treat the return value as "the set of queries to retrieve for".
    """
    if not config.ENABLE_QUERY_DECOMPOSITION:
        return [question]
    if not _heuristically_needs_decomposition(question):
        return [question]

    prompt = _DECOMPOSITION_INSTRUCTION.format(
        max_subqueries=config.MAX_SUBQUERIES, question=question
    )

    try:
        raw = _call_gemini_decomposition(prompt)
        cleaned = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        sub_questions = json.loads(cleaned)
        if not isinstance(sub_questions, list) or not sub_questions:
            return [question]
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
        return sub_questions[: config.MAX_SUBQUERIES] or [question]
    except Exception:
        logger.exception("Query decomposition failed for %r — falling back to single query.", question)
        return [question]


# =====================================================================
# Multimodal figure summarization
#
# Cropped figures are currently only findable if the question mentions
# their exact label ("Figure 8-2"). We generate a short, searchable text
# description of each figure at indexing time (schematic elements, what
# it shows) and embed that description as its own chunk, so a question
# like "how is flash connected?" can retrieve the schematic even though
# it never names the figure.
# =====================================================================

_FIGURE_SUMMARY_INSTRUCTION = """This image is a figure cropped from an electronics \
datasheet, labeled "{label}".{caption_context} In 2-4 sentences, describe what it shows \
for someone searching a datasheet — e.g. whether it's a schematic, pinout diagram, timing \
diagram, graph, or package drawing, and the key labeled elements, signals, or components \
visible (list specific names/labels you can read, such as pin names, component references, \
or axis labels). Be factual and only describe what is visibly present. Do not speculate \
about anything not shown. Respond with the description only, no preamble."""


def _figure_cache_conn():
    conn = sqlite3.connect(config.FIGURE_SUMMARY_CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS figure_summaries (hash TEXT PRIMARY KEY, summary TEXT NOT NULL)"
    )
    return conn


def _figure_cache_key(image_path: str, label: str, caption: str) -> str:
    raw = f"{config.FIGURE_SUMMARY_MODEL}|{label}|{caption}|{image_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _figure_cache_get(key: str) -> str | None:
    conn = _figure_cache_conn()
    try:
        row = conn.execute(
            "SELECT summary FROM figure_summaries WHERE hash = ?", (key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _figure_cache_set(key: str, summary: str):
    conn = _figure_cache_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO figure_summaries (hash, summary) VALUES (?, ?)",
            (key, summary),
        )
        conn.commit()
    finally:
        conn.close()


@retry(
    stop=stop_after_attempt(config.API_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.API_BACKOFF_BASE_SECONDS, min=1, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini_vision(prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str:
    response = _genai_client.models.generate_content(
        model=config.FIGURE_SUMMARY_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text


def summarize_figure(image_path: str, label: str, caption: str = "") -> str | None:
    """
    Returns a short searchable description of a cropped figure image,
    cached on disk keyed by (model, label, caption, path). Returns None if
    summarization is disabled or fails — callers should skip creating a
    figure-summary chunk in that case rather than error out.
    """
    if not config.ENABLE_FIGURE_SUMMARIES:
        return None

    key = _figure_cache_key(image_path, label, caption)
    cached = _figure_cache_get(key)
    if cached is not None:
        return cached

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        caption_context = f' Its printed caption reads: "{caption}"' if caption else ""
        prompt = _FIGURE_SUMMARY_INSTRUCTION.format(label=label, caption_context=caption_context)
        summary = _call_gemini_vision(prompt, image_bytes)
        summary = summary.strip()[: config.FIGURE_SUMMARY_MAX_CHARS]
        _figure_cache_set(key, summary)
        return summary
    except Exception:
        logger.exception("Figure summarization failed for %s (%s)", label, image_path)
        return None


# =====================================================================
# Structured table extraction fallback
#
# pdfplumber's line/text-clustering strategies mangle merged cells and
# multi-page tables (e.g. register maps). When a geometrically-extracted
# table looks low-quality, we fall back to asking the model to read the
# cropped table region directly and emit clean structured rows.
# =====================================================================

_TABLE_STRUCTURING_INSTRUCTION = """This image is a table cropped from an electronics \
datasheet. Read it carefully, including merged cells (repeat the merged value into each \
row/column it spans) and any multi-row headers (flatten them into a single descriptive \
header per column). Respond with ONLY a JSON object of this exact shape, nothing else:

{{"headers": ["col1", "col2", ...], "rows": [["val1", "val2", ...], ...]}}

If a cell is empty, use an empty string. Preserve the original text of each cell exactly \
as printed (including units, symbols, min/typ/max labels)."""


@retry(
    stop=stop_after_attempt(config.API_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.API_BACKOFF_BASE_SECONDS, min=1, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini_table_vision(image_bytes: bytes, mime_type: str = "image/png") -> str:
    response = _genai_client.models.generate_content(
        model=config.TABLE_STRUCTURING_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            _TABLE_STRUCTURING_INSTRUCTION,
        ],
    )
    return response.text


def structure_table_from_image(image_bytes: bytes) -> dict | None:
    """
    Asks the model to read a cropped table image and return
    {"headers": [...], "rows": [[...], ...]}. Returns None on failure.
    """
    if not config.ENABLE_STRUCTURED_TABLE_FALLBACK:
        return None

    try:
        raw = _call_gemini_table_vision(image_bytes)
        cleaned = re.sub(r"^```json|^```|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict) or "headers" not in parsed or "rows" not in parsed:
            return None
        if not parsed["headers"] or not isinstance(parsed["rows"], list):
            return None
        return parsed
    except Exception:
        logger.exception("Structured table extraction failed.")
        return None