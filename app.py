"""
app.py — Streamlit frontend for the Datasheet Assistant.
"""

import os
import json
import tempfile
import logging
import re

import pandas as pd
import streamlit as st

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Datasheet Assistant", page_icon="📘", layout="wide")

if not config.GEMINI_API_KEY:
    st.error(
        "No GEMINI_API_KEY found. Set it as an environment variable or in a .env file "
        "before running the app (see .env.example). Get a free key at "
        "https://aistudio.google.com/apikey"
    )
    st.stop()

# Imported only after the key check above: backend.generator and
# backend.vectorspace both construct a Gemini client at import time, so
# importing them first would fail with a raw SDK traceback on a missing
# key instead of the friendly message above.
from backend import pdf, chunking, vectorspace, generator

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, images}


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("📘 Datasheet Assistant")
    st.caption("Upload datasheets, then ask questions grounded in the actual pages.")

    uploaded_files = st.file_uploader(
        "Upload datasheet PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process & Index", use_container_width=True):
        progress = st.progress(0.0, text="Starting...")
        indexed_count = 0
        failed_files = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                i / len(uploaded_files),
                text=f"Processing {uploaded_file.name}...",
            )
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                page_records = pdf.process_pdf(tmp_path, uploaded_file.name)
                chunks = chunking.build_chunks(page_records)
                vectorspace.add_chunks(chunks)
                indexed_count += 1
            except Exception:
                logger.exception("Failed to index %s", uploaded_file.name)
                failed_files.append(uploaded_file.name)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        progress.progress(1.0, text="Done.")
        if indexed_count:
            st.success(f"Indexed {indexed_count} document(s).")
        if failed_files:
            st.warning(f"Could not process: {', '.join(failed_files)}. Check they are valid PDFs.")

    st.divider()
    indexed_docs = vectorspace.list_indexed_documents()
    if indexed_docs:
        st.subheader("Indexed documents")
        for doc in indexed_docs:
            st.markdown(f"- {doc}")
        if st.button("Clear all documents", use_container_width=True):
            vectorspace.clear_all()
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF to get started.")


# ------------------------------------------------------------- Main chat
st.header("Ask about your datasheets")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tables"):
            for t in msg["tables"]:
                st.markdown(f"*{t['document']} — Page {t['page']}*")
                st.dataframe(pd.DataFrame(t["records"], columns=t["columns"]), use_container_width=True)
        if msg.get("images"):
            cols = st.columns(min(len(msg["images"]), 3))
            for idx, img in enumerate(msg["images"]):
                with cols[idx % len(cols)]:
                    caption = f"{img['document']} — Page {img['page']}"
                    if img.get("label"):
                        caption += f" ({img['label']})"
                    st.image(img["path"], caption=caption)

question = st.chat_input("e.g. What is the maximum input voltage? / Show me Figure 6")

if question:
    if not indexed_docs:
        st.warning("Please upload and index at least one datasheet first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching datasheets..."):
            try:
                direct_ref = generator.extract_direct_reference(question)
                if direct_ref:
                    hits = vectorspace.find_by_figure_or_table(direct_ref)
                    if not hits:
                        hits = vectorspace.query(question)
                else:
                    sub_questions = generator.decompose_question(question)
                    if len(sub_questions) > 1:
                        st.caption("🔎 Searching sub-topics: " + " · ".join(sub_questions))
                    hits = vectorspace.multi_query(sub_questions)

                answer = generator.generate_answer(question, hits)
            except Exception:
                logger.exception("Failed to answer question: %s", question)
                answer = ("Something went wrong while generating an answer. "
                          "Please try again in a moment.")
                hits = []

            # --- Refinement 6: Citation Grounding UI Filter ---
            raw_citations = re.findall(r"\[([^,\]]+?\.pdf),\s*(?:Page\s*)?(\d+)\]", answer, re.IGNORECASE)
            cited_pages = {(doc.strip(), int(pg)) for doc, pg in raw_citations}

            seen_pages = set()
            images = []
            for hit in hits:
                meta = hit["metadata"]
                doc_name = meta["document"]
                page_num = int(meta["page"])
                
                if cited_pages and (doc_name, page_num) not in cited_pages:
                    continue

                key = (doc_name, page_num)
                if key in seen_pages:
                    continue
                seen_pages.add(key)

                crop_path = None
                if direct_ref:
                    figure_crops = json.loads(meta.get("figure_crops") or "{}")
                    crop_path = figure_crops.get(direct_ref)

                images.append({
                    "path": crop_path or meta["image_path"],
                    "document": doc_name,
                    "page": page_num,
                    "label": direct_ref if crop_path else None,
                })

            tables = []
            for hit in hits:
                meta = hit["metadata"]
                doc_name = meta["document"]
                page_num = int(meta["page"])

                if cited_pages and (doc_name, page_num) not in cited_pages:
                    continue

                raw_rows = meta.get("table_rows")
                if not raw_rows:
                    continue
                rows = json.loads(raw_rows)
                if len(rows) < 2:
                    continue
                
                # --- Refinement 5: Text length header truncation fallback ---
                raw_headers = rows[0]
                clean_headers = []
                for idx, col in enumerate(raw_headers):
                    col_str = str(col).strip() if col else ""
                    col_str = re.sub(r"\s+", " ", col_str)
                    
                    if len(col_str) > 30:
                        col_str = f"Column_{idx}"
                    elif not col_str:
                        col_str = f"EmptyCol_{idx}"
                    elif col_str in clean_headers:
                        col_str = f"{col_str}_{idx}"
                        
                    clean_headers.append(col_str)

                tables.append({
                    "document": doc_name,
                    "page": page_num,
                    "df": pd.DataFrame(rows[1:], columns=clean_headers),
                })

        st.markdown(answer)

        if tables:
            st.caption("📊 Source tables:")
            for t in tables:
                st.markdown(f"*{t['document']} — Page {t['page']}*")
                st.dataframe(t["df"], use_container_width=True)

        if images:
            st.caption("📄 Source pages (grounding for this answer):")
            cols = st.columns(min(len(images), 3))
            for idx, img in enumerate(images):
                with cols[idx % len(cols)]:
                    caption = f"{img['document']} — Page {img['page']}"
                    if img["label"]:
                        caption += f" ({img['label']})"
                    st.image(img["path"], caption=caption)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "images": images,
        "tables": [{"document": t["document"], "page": t["page"], "records": t["df"].to_dict("records"), "columns": list(t["df"].columns)} for t in tables],
    })