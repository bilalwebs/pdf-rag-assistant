import os
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF, build a FAISS index, and ask questions using an open-source model served by Groq.")

# -----------------------------
# Configuration
# -----------------------------
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((page_number, text))

    return pages


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page_number, text in pages:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def build_faiss_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve(query, index, chunks, model, top_k=TOP_K):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


def generate_answer(question, retrieved_chunks, client):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful PDF question-answering assistant.
Answer the user's question using ONLY the supplied document context.
If the answer is not present in the context, say:
"I couldn't find the answer in the uploaded document."
Do not invent facts.
When possible, mention the relevant page number(s).
Keep the answer clear and concise."""

    user_prompt = f"""Document context:
{context}

Question:
{question}"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE}` characters")
    st.write(f"**Chunk overlap:** `{CHUNK_OVERLAP}`")
    st.write(f"**Retrieved chunks:** `{TOP_K}`")

    if st.button("🗑️ Clear document"):
        for key in ["index", "chunks", "file_name"]:
            st.session_state.pop(key, None)
        st.rerun()


# -----------------------------
# API key check
# -----------------------------
client = get_groq_client()

if client is None:
    st.warning("GROQ_API_KEY is not configured.")
    st.info(
        "For local development, set GROQ_API_KEY as an environment variable "
        "or add it to .streamlit/secrets.toml. On Streamlit Cloud, add it under "
        "App Settings → Secrets."
    )


# -----------------------------
# PDF upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned/image-only PDFs require OCR.",
)

if uploaded_file is not None and client is not None:
    if (
        st.session_state.get("file_name") != uploaded_file.name
        or "index" not in st.session_state
    ):
        with st.spinner("Reading PDF and building the vector index..."):
            pages = extract_pdf_text(uploaded_file)

            if not pages:
                st.error(
                    "No extractable text was found. This may be a scanned/image-only PDF. "
                    "OCR support is not included in this starter version."
                )
                st.stop()

            chunks = chunk_text(pages)

            if not chunks:
                st.error("The PDF did not produce any usable text chunks.")
                st.stop()

            embedding_model = load_embedding_model()
            index, _ = build_faiss_index(chunks, embedding_model)

            st.session_state["index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["file_name"] = uploaded_file.name

        st.success(
            f"Indexed **{len(pages)} pages** into **{len(chunks)} chunks**."
        )

    else:
        st.success(
            f"Using indexed document: **{st.session_state['file_name']}** "
            f"({len(st.session_state['chunks'])} chunks)"
        )

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        embedding_model = load_embedding_model()

        with st.spinner("Searching the document..."):
            retrieved = retrieve(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                embedding_model,
                TOP_K,
            )

        with st.spinner("Generating answer with Groq..."):
            answer = generate_answer(question, retrieved, client)

        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)

        with st.expander("🔎 Retrieved context"):
            for i, item in enumerate(retrieved, start=1):
                st.markdown(
                    f"**Chunk {i} — Page {item['page']} — "
                    f"Similarity: {item['score']:.3f}**"
                )
                st.write(item["text"])
                st.divider()

elif uploaded_file is None:
    st.info("Upload a PDF above to start chatting with your document.")
