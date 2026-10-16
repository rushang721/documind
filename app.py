import os

import streamlit as st

from answer import ask
from embed import embed_texts
from ingest import read_pdf, split_pages
from retrieve import top_chunks

st.set_page_config(page_title="DocuMind", page_icon="📚")
st.title("DocuMind")
st.caption("Ask questions across your PDFs. Every answer shows where it came from.")


def default_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


api_key = st.sidebar.text_input("Gemini API key", type="password", value=default_key())
files = st.sidebar.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
top_k = st.sidebar.slider("Excerpts per question", 3, 8, 5)

signature = tuple((f.name, f.size) for f in files)
if files and api_key and st.session_state.get("signature") != signature:
    with st.spinner("Reading and indexing..."):
        chunks = []
        for f in files:
            chunks += split_pages(read_pdf(f), f.name)
        if not chunks:
            st.sidebar.error("No text found. Scanned PDFs need OCR first.")
        else:
            try:
                matrix = embed_texts([c.text for c in chunks], api_key)
            except Exception as err:
                st.sidebar.error(f"Indexing failed: {err}")
            else:
                st.session_state.update(
                    signature=signature, chunks=chunks, matrix=matrix, messages=[]
                )

if "chunks" in st.session_state:
    st.sidebar.success(
        f"{len(files)} file(s), {len(st.session_state.chunks)} chunks indexed"
    )

for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("sources"):
            with st.expander("Sources"):
                for c, score in m["sources"]:
                    st.caption(f"{c.doc}, p.{c.page} (similarity {score:.2f})")
                    st.write(c.text[:350] + "...")

if question := st.chat_input("Ask about your documents"):
    if "chunks" not in st.session_state:
        st.warning("Add your API key and upload a PDF first.")
    else:
        history = [(m["role"], m["text"]) for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "text": question})
        try:
            hits = top_chunks(
                question, st.session_state.chunks, st.session_state.matrix, api_key, top_k
            )
            reply = ask(question, hits, history, api_key)
        except Exception as err:
            reply, hits = f"Something went wrong: {err}", []
        st.session_state.messages.append(
            {"role": "assistant", "text": reply, "sources": hits}
        )
        st.rerun()
