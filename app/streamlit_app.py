import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import load_and_chunk
from src.embeddings import load_embedding_model
from src.vector_store import add_documents, clear_vector_store
from src.rag_chain import ask_V2
import tempfile

st.set_page_config(
    page_title="Study Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0d0d0d;
    --bg-secondary: #141414;
    --bg-tertiary: #1a1a1a;
    --border: #222222;
    --text-primary: #f0ede8;
    --text-secondary: #888888;
    --text-muted: #444444;
    --accent: #e8a030;
    --accent-dim: #3a2800;
}

* { font-family: 'JetBrains Mono', monospace; }

.stApp { background-color: var(--bg); color: var(--text-primary); }

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * { color: var(--text-primary); }

.stTextInput > div > div > input {
    background-color: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 2px !important;
    font-size: 0.8em !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
}

.stButton > button {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75em !important;
    letter-spacing: 0.05em !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: var(--accent-dim) !important;
}

div[data-testid="stFileUploader"] {
    background-color: var(--bg-tertiary);
    border: 1px dashed var(--border);
    border-radius: 2px;
}

.stDivider { border-color: var(--border) !important; }

.stChatInput > div {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}

.stChatInput textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
    color: var(--text-primary) !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1em;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    font-size: 0.7em;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0;
}

.section-label {
    font-size: 0.65em;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.doc-pill {
    display: inline-block;
    background-color: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 0.65em;
    letter-spacing: 0.05em;
    margin: 2px 2px 2px 0;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.user-bubble {
    background-color: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 14px 18px;
    margin: 12px 0 4px 0;
    font-size: 0.85em;
    color: var(--text-primary);
    line-height: 1.6;
}

.user-label {
    font-size: 0.6em;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.assistant-bubble {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 2px 2px 0;
    padding: 14px 18px;
    margin: 4px 0 4px 0;
    font-size: 0.85em;
    color: var(--text-primary);
    line-height: 1.7;
}

.assistant-label {
    font-size: 0.6em;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.source-block {
    border-left: 2px solid var(--accent-dim);
    padding: 6px 12px;
    margin-top: 10px;
    font-size: 0.7em;
    color: var(--text-muted);
    letter-spacing: 0.03em;
}

.source-block span {
    color: var(--accent);
}

.empty-state {
    text-align: center;
    padding: 120px 0;
    color: var(--text-muted);
    font-size: 0.75em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4em;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

# Sidebar
with st.sidebar:
    st.markdown('<p class="sidebar-title">Study Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-subtitle">RAG — powered by Llama 3</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "upload",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    url_input = st.text_input(
        "url",
        placeholder="https://",
        label_visibility="collapsed"
    )

    if st.button("Load", use_container_width=True):
        sources = []
        embedding_model = get_embedding_model()

        if uploaded_files:
            for file in uploaded_files:
                suffix = ".pdf" if file.type == "application/pdf" else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                source_type = "pdf" if suffix == ".pdf" else "txt"
                sources.append({"type": source_type, "path": tmp_path, "name": file.name})

        if url_input.strip():
            sources.append({"type": "url", "path": url_input.strip(), "name": url_input.strip()})

        if sources:
            with st.spinner("processing..."):
                for source in sources:
                    chunks = load_and_chunk(
                        [{"type": source["type"], "path": source["path"]}],
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    add_documents(chunks, get_embedding_model())
                    if source["name"] not in st.session_state.loaded_docs:
                        st.session_state.loaded_docs.append(source["name"])
            st.success(f"{len(sources)} source(s) loaded")
        else:
            st.warning("no documents provided")

    if st.session_state.loaded_docs:
        st.divider()
        st.markdown('<p class="section-label">Loaded</p>', unsafe_allow_html=True)
        for doc in st.session_state.loaded_docs:
            name = doc if len(doc) < 30 else doc[:27] + "..."
            st.markdown(f'<span class="doc-pill">{name}</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("Clear Session", use_container_width=True):
        clear_vector_store()
        st.session_state.messages = []
        st.session_state.loaded_docs = []
        st.rerun()

# Main area
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    if not st.session_state.loaded_docs:
        st.markdown("""
        <div class="empty-state">
            upload your notes to begin
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <div class="user-label">you</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-bubble">
                    <div class="assistant-label">assistant</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                if message.get("sources"):
                    sources_str = " · ".join([f"<span>{s}</span>" for s in message["sources"]])
                    st.markdown(f'<div class="source-block">sources — {sources_str}</div>', unsafe_allow_html=True)

        question = st.chat_input("ask something...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner(""):
                from src.vector_store import similarity_search
                answer = ask_V2(question, embedding_model=get_embedding_model(), k=4)
                retrieved_docs = similarity_search(question, k=4, embedding_model=get_embedding_model())
                sources = list(set([
                    doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
                    for doc in retrieved_docs
                ]))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            st.rerun()