"""
Streamlit frontend for the Farmers Assistant Chatbot.

Run with:   streamlit run app.py

This file only handles the user interface. All document processing,
searching and answering happens in backend.py.
"""

import re

import streamlit as st

import backend


st.set_page_config(page_title="Farmers Assistant Chatbot", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------------------
# Load the backend once. st.cache_resource keeps the same object between
# Streamlit reruns, so the embedding model and index are not reloaded every
# time the user clicks something.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the embedding model and document index...")
def get_knowledge_base() -> backend.KnowledgeBase:
    return backend.KnowledgeBase()


@st.cache_data(ttl=30, show_spinner=False)
def get_llm_status() -> tuple[bool, str]:
    return backend.check_llm()


def escape_markdown(text: str) -> str:
    """Stop filenames or PDF text from being interpreted as Markdown/LaTeX."""
    return re.sub(r"([\\`*_{}\[\]<>#|$~])", r"\\\1", text)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)})"):
        for source in sources:
            st.markdown(f"**[{source['number']}] {escape_markdown(source['citation'])}**  "
                        f"\n_Relevance score: {source['score']:.2f}_")
            st.text(source["text"][:1200])
            st.divider()


def show_assistant_message(message: dict) -> None:
    status = message["status"]
    answer = message["answer"]
    if status == "llm_error":
        st.error(answer)
    elif status in ("insufficient", "no_documents"):
        st.warning(answer)
        if status == "insufficient":
            st.caption("Tip: answers come only from the documents in the knowledge base. "
                       "Try rephrasing the question, or upload a PDF that covers this topic.")
    else:
        st.markdown(answer.replace("$", r"\$"))
        st.caption("AI-generated summary of the sources below. Please check it against "
                   "the source text before acting on it.")
    show_sources(message.get("sources", []))


kb = get_knowledge_base()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_results" not in st.session_state:
    st.session_state.upload_results = []
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None
if "notice" not in st.session_state:
    st.session_state.notice = None


# ---------------------------------------------------------------------------
# Sidebar: uploads, document list, settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📄 Knowledge base")

    # --- Upload ---
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            f"Upload PDF documents (max {backend.MAX_UPLOAD_MB} MB each)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Add to knowledge base", type="primary")

    if submitted:
        if not uploaded_files:
            st.session_state.upload_results = [
                {"status": "error", "message": "Please choose at least one PDF file first."}]
        else:
            results = []
            for uploaded in uploaded_files:
                with st.status(f"Processing {uploaded.name}...", expanded=False) as status:
                    status.write("Checking file, extracting text, creating embeddings...")
                    result = kb.add_pdf(uploaded.getvalue(), uploaded.name)
                    state = {"added": "complete", "duplicate": "complete"}.get(result["status"], "error")
                    status.update(label=uploaded.name, state=state)
                results.append(result)
            st.session_state.upload_results = results

    for result in st.session_state.upload_results:
        if result["status"] == "added":
            st.success(result["message"])
        elif result["status"] == "duplicate":
            st.info(result["message"])
        else:
            st.error(result["message"])

    if st.session_state.notice:
        st.success(st.session_state.notice)
        st.session_state.notice = None

    # --- Document list ---
    documents = kb.list_documents()
    st.subheader(f"Available documents ({len(documents)})")

    if not documents:
        st.info("No documents yet. Upload a PDF to get started.")

    for doc in documents:
        with st.container(border=True):
            if doc["source_type"] == "original":
                st.markdown("**🌾 Original rice collection**")
                st.caption(f"From data/chunks.txt · {doc['chunk_count']} chunks · "
                           "page numbers not available")
            else:
                st.markdown(f"**📘 {escape_markdown(doc['display_name'])}**")
                st.caption(f"Uploaded · {doc['pages']} pages · {doc['chunk_count']} chunks")

            if st.session_state.confirm_delete == doc["doc_id"]:
                st.warning("Remove this document from the knowledge base?")
                col_yes, col_no = st.columns(2)
                if col_yes.button("Yes, remove", key=f"yes_{doc['doc_id']}", type="primary"):
                    st.session_state.notice = kb.remove_document(doc["doc_id"])
                    st.session_state.confirm_delete = None
                    st.session_state.upload_results = []
                    st.rerun()
                if col_no.button("Cancel", key=f"no_{doc['doc_id']}"):
                    st.session_state.confirm_delete = None
                    st.rerun()
            elif st.button("Remove", key=f"remove_{doc['doc_id']}"):
                st.session_state.confirm_delete = doc["doc_id"]
                st.rerun()

    has_original = any(d["source_type"] == "original" for d in documents)
    if backend.ORIGINAL_CHUNKS_FILE.exists():
        label = "Re-import data/chunks.txt" if has_original else "Restore original rice collection"
        if st.button(label, help="Rebuilds the original collection from data/chunks.txt."):
            with st.spinner("Importing data/chunks.txt..."):
                st.session_state.notice = kb.import_original_chunks()
            st.rerun()

    # --- Settings and status ---
    st.divider()
    with st.expander("⚙️ Retrieval settings"):
        min_similarity = st.slider(
            "Minimum relevance score", 0.0, 1.0, backend.DEFAULT_MIN_SIMILARITY, 0.05,
            help="Passages scoring below this are ignored. Higher = stricter. "
                 "A high score means similar wording, not guaranteed correctness.")
        top_k = st.slider("Passages to retrieve", 1, 8, backend.DEFAULT_TOP_K)

    llm_ok, llm_message = get_llm_status()
    if llm_ok:
        st.caption(f"🟢 {llm_message}")
    else:
        st.error(llm_message)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area: title, description and chat
# ---------------------------------------------------------------------------

st.title("🌾 Farmers Assistant Chatbot")
st.markdown(
    "Ask questions about farming and get answers taken **only from the documents "
    "in the knowledge base**, with the source of each answer. The assistant starts "
    "with rice cultivation, pest and disease documents, and you can upload more PDFs "
    "from the sidebar."
)
st.caption("Answers are only as good as the uploaded documents. Always confirm pesticide "
           "or fertiliser decisions with your local agricultural officer.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"])
        else:
            show_assistant_message(message)

question = st.chat_input(
    "Ask a question about your documents..." if documents else "Upload a document to start chatting",
    max_chars=500,
    disabled=not documents,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.text(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and preparing an answer..."):
            result = kb.answer(question, min_similarity=min_similarity, top_k=top_k)
        message = {
            "role": "assistant",
            "answer": result.answer,
            "status": result.status,
            "sources": result.sources,
        }
        show_assistant_message(message)

    st.session_state.messages.append(message)
