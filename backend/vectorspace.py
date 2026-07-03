"""
backend/vectorstore.py

Wraps ChromaDB for storage/retrieval, using Gemini's embedding model.
Includes Hybrid search (Vector + BM25) with Reciprocal Rank Fusion (RRF),
alphanumeric regex tokenization, de-duplicated set expansions, and dynamic section scaling.
"""

import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict

import chromadb
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger(__name__)

config.require_gemini_key()
_genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
_client = chromadb.PersistentClient(path=config.VECTORSTORE_DIR)
_collection = _client.get_or_create_collection(name="datasheets")

_bm25_index: BM25Okapi | None = None
_bm25_ids: list[str] = []
_bm25_corpus_dirty = True

# Datasheet section headings appear in whatever case the source PDF uses
# (often ALL CAPS in older TI/ST sheets), so match SECTION_BOOST_MAP
# case-insensitively rather than requiring an exact-case key match.
_SECTION_BOOST_MAP_CI = {name.lower(): keywords for name, keywords in config.SECTION_BOOST_MAP.items()}


def _tokenize(text: str) -> list[str]:
    """
    Advanced Alphanumeric Tokenizer.
    Preserves specialized microelectronics tokens such as '3.3V', 'CHIP_EN', 'GPIO0'.
    """
    return re.findall(r"[a-z0-9_+\-/.]+", text.lower())


def _rebuild_bm25_index():
    global _bm25_index, _bm25_ids, _bm25_corpus_dirty

    all_data = _collection.get(include=["documents"])
    ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])

    if not documents:
        _bm25_index = None
        _bm25_ids = []
        _bm25_corpus_dirty = False
        return

    tokenized_corpus = [_tokenize(doc) for doc in documents]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_ids = ids
    _bm25_corpus_dirty = False


def _ensure_bm25_fresh():
    if _bm25_corpus_dirty or _bm25_index is None:
        _rebuild_bm25_index()


# --- Embedding cache (SQLite) ----------------------

def _cache_conn():
    conn = sqlite3.connect(config.EMBEDDING_CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, vector TEXT NOT NULL)"
    )
    return conn


def _cache_key(text: str, task_type: str) -> str:
    raw = f"{config.EMBEDDING_MODEL}|{task_type}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> list[float] | None:
    conn = _cache_conn()
    try:
        row = conn.execute("SELECT vector FROM embeddings WHERE hash = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()


def _cache_set(key: str, vector: list[float]):
    conn = _cache_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (hash, vector) VALUES (?, ?)",
            (key, json.dumps(vector)),
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
def _embed_via_api(text: str, task_type: str) -> list[float]:
    result = _genai_client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type.upper()),
    )
    return result.embeddings[0].values


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    key = _cache_key(text, task_type)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    vector = _embed_via_api(text, task_type)
    _cache_set(key, vector)
    return vector


def add_chunks(chunks: list[dict]):
    global _bm25_corpus_dirty

    if not chunks:
        return

    ids, embeddings, documents, metadatas = [], [], [], []

    for chunk in chunks:
        ids.append(chunk["id"])
        embeddings.append(embed_text(chunk["text"], task_type="retrieval_document"))
        documents.append(chunk["text"])
        metadatas.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "section": chunk.get("section", ""),
            "category": chunk.get("category", "General"),
            "keywords": ", ".join(chunk.get("keywords", [])),
            "manufacturer": chunk.get("manufacturer", "Unknown"),
            "component": chunk.get("component", "Unknown"),
            "revision": chunk.get("revision", "Unknown"),
            "figures": ", ".join(chunk["figures"]),
            "tables_refs": ", ".join(chunk["tables_refs"]),
            "image_path": chunk["image_path"],
            "figure_crops": json.dumps(chunk.get("figure_crops", {})),
            "table_rows": json.dumps(chunk["table_rows"]) if chunk.get("table_rows") else "",
        })

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    _bm25_corpus_dirty = True


def _vector_candidates(question: str, pool_size: int) -> list[tuple[str, dict, str]]:
    query_embedding = embed_text(question, task_type="retrieval_query")
    results = _collection.query(query_embeddings=[query_embedding], n_results=pool_size)

    ranked = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            ranked.append((
                results["ids"][0][i],
                results["metadatas"][0][i],
                results["documents"][0][i],
            ))
    return ranked


def _bm25_candidates(question: str, pool_size: int) -> list[str]:
    _ensure_bm25_fresh()
    if _bm25_index is None:
        return []

    scores = _bm25_index.get_scores(_tokenize(question))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [_bm25_ids[i] for i in ranked_indices[:pool_size] if scores[i] > 0]


def _reciprocal_rank_fusion(
    vector_ranked_ids: list[str],
    bm25_ranked_ids: list[str],
    k: int = None,
) -> dict[str, float]:
    k = k or config.RRF_K
    fused: dict[str, float] = defaultdict(float)

    for rank, doc_id in enumerate(vector_ranked_ids):
        fused[doc_id] += config.VECTOR_WEIGHT / (k + rank + 1)

    for rank, doc_id in enumerate(bm25_ranked_ids):
        fused[doc_id] += config.BM25_WEIGHT / (k + rank + 1)

    return fused


def _expand_query(question: str) -> str:
    """
    Compiles de-duplicated synonym arrays using Python sets, preventing nested 
    duplication pipelines from skewing BM25 internal frequency factors.

    Expands on two levels:
      - single-word triggers (config.QUERY_SYNONYMS), matched by word boundary
      - multi-word phrase triggers (config.PHRASE_SYNONYMS), matched by
        substring search, since terms like "minimum circuit" or "reference
        design" carry meaning as a phrase that word-level lookup would miss
        (neither "minimum" nor "circuit" alone implies "schematic").
    """
    lowered = question.lower()
    expansions = set()

    for key, synonyms in config.QUERY_SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            expansions.update(synonyms)

    for phrase, synonyms in config.PHRASE_SYNONYMS.items():
        if phrase in lowered:
            expansions.update(synonyms)

    if expansions:
        return f"{question} {' '.join(sorted(expansions))}"
    return question


def query(question: str, top_k: int = None) -> list[dict]:
    """
    Hybrid search incorporating alphanumeric token tracking, clean unique set expansion routines, 
    and smooth keyword matching coefficients.
    """
    top_k = top_k or config.TOP_K_RESULTS
    pool_size = max(config.CANDIDATE_POOL_SIZE, top_k)

    # 1. Expansion
    expanded_question = _expand_query(question)

    # 2. Dual Search
    vector_hits = _vector_candidates(expanded_question, pool_size)
    vector_ranked_ids = [hit[0] for hit in vector_hits]
    lookup = {hit[0]: (hit[1], hit[2]) for hit in vector_hits}

    bm25_ranked_ids = _bm25_candidates(expanded_question, pool_size)

    # 3. Fusion
    fused_scores = _reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids)
    if not fused_scores:
        return []

    missing_ids = [doc_id for doc_id in fused_scores if doc_id not in lookup]
    if missing_ids:
        fetched = _collection.get(ids=missing_ids, include=["documents", "metadatas"])
        for doc_id, meta, text in zip(fetched["ids"], fetched["metadatas"], fetched["documents"]):
            lookup[doc_id] = (meta, text)

    # 4. Smooth Structural Intersections Scaling
    lowered_question = question.lower()
    for doc_id, (meta, text) in lookup.items():
        section = meta.get("section", "")
        target_keywords = _SECTION_BOOST_MAP_CI.get(section.lower())
        if target_keywords:
            # Count distinct structural intersections
            match_count = sum(1 for kw in target_keywords if re.search(rf"\b{re.escape(kw)}\b", lowered_question))

            if match_count > 0:
                # Smooth scaling equation: score * (1 + 0.05 * matches) capped at MAX_SECTION_MULTIPLIER
                dynamic_multiplier = min(1.0 + (config.BASE_BOOST_PER_MATCH * match_count), config.MAX_SECTION_MULTIPLIER)
                fused_scores[doc_id] *= dynamic_multiplier

    # 5. Output — page-diverse selection.
    # Selecting the raw top_k by score alone tends to return several chunks
    # from the same heavily-matched page (e.g. 5 chunks from page 22) at the
    # expense of other relevant pages (e.g. the schematic on page 32) never
    # appearing at all. Walk the ranked list and cap how many chunks from
    # the same page can be selected, then backfill with the highest-scoring
    # leftovers if the cap left the result set short of top_k.
    all_ranked_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)

    selected_ids = []
    page_counts: dict[tuple, int] = defaultdict(int)
    leftover_ids = []

    for doc_id in all_ranked_ids:
        if len(selected_ids) >= top_k:
            break
        meta, _ = lookup[doc_id]
        page_key = (meta.get("document", ""), meta.get("page", ""))
        if page_counts[page_key] < config.MAX_CHUNKS_PER_PAGE:
            selected_ids.append(doc_id)
            page_counts[page_key] += 1
        else:
            leftover_ids.append(doc_id)

    if len(selected_ids) < top_k:
        selected_ids.extend(leftover_ids[: top_k - len(selected_ids)])

    hits = []
    for doc_id in selected_ids:
        meta, text = lookup[doc_id]
        hits.append({"id": doc_id, "text": text, "metadata": meta, "score": fused_scores[doc_id]})
    return hits


def multi_query(sub_questions: list[str], top_k_each: int = None, merged_top_k: int = None) -> list[dict]:
    """
    Runs query() independently for each sub-question (from query
    decomposition) and merges the results.

    A chunk retrieved by multiple sub-questions keeps its best score
    rather than being summed, so one sub-question's high-confidence hit
    isn't diluted by weaker hits for the same chunk under other
    sub-questions; the merge is about maximizing recall across topics,
    not re-ranking by aggregate relevance.
    """
    top_k_each = top_k_each or config.TOP_K_PER_SUBQUERY
    merged_top_k = merged_top_k or config.TOP_K_RESULTS

    if len(sub_questions) <= 1:
        return query(sub_questions[0] if sub_questions else "", top_k=merged_top_k)

    best_by_id: dict[str, dict] = {}
    order: list[str] = []  # preserves first-seen order as a stable tiebreaker

    for sub_q in sub_questions:
        for hit in query(sub_q, top_k=top_k_each):
            doc_id = hit["id"]
            existing = best_by_id.get(doc_id)
            if existing is None:
                best_by_id[doc_id] = hit
                order.append(doc_id)
            elif (hit["score"] or 0) > (existing["score"] or 0):
                best_by_id[doc_id] = hit

    ranked = sorted(
        (best_by_id[doc_id] for doc_id in order),
        key=lambda h: h["score"] or 0,
        reverse=True,
    )
    return ranked[:merged_top_k]


def find_by_figure_or_table(reference: str) -> list[dict]:
    all_data = _collection.get(include=["documents", "metadatas"])
    hits = []
    for doc_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        combined_refs = f"{meta.get('figures', '')} {meta.get('tables_refs', '')}"
        if reference.lower() in combined_refs.lower():
            hits.append({"id": doc_id, "text": text, "metadata": meta, "score": None})
    return hits


def list_indexed_documents() -> list[str]:
    all_data = _collection.get(include=["metadatas"])
    docs = {m["document"] for m in all_data["metadatas"]}
    return sorted(docs)


def clear_all():
    global _collection, _bm25_corpus_dirty
    _client.delete_collection("datasheets")
    _collection = _client.get_or_create_collection(name="datasheets")
    _bm25_corpus_dirty = True