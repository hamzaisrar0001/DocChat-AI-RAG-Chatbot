import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from datetime import datetime
import tempfile
import time
import os

load_dotenv()
HF_TOKEN = os.getenv("CHROMA_HUGGINGFACE_API_KEY") or st.secrets.get("CHROMA_HUGGINGFACE_API_KEY", None)

st.set_page_config(
    page_title="DocChat AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #fafafa;
    font-family: 'Sora', sans-serif;
    color: #1a1a2e;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ffe0c2;
}

[data-testid="stSidebar"] > div:first-child { padding: 1.8rem 1.4rem; }

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #ffe0c2;
}

.brand-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #ff6b00, #ff9a3c);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 14px rgba(255,107,0,0.25);
}

.brand-name { font-size: 1.1rem; font-weight: 700; color: #1a1a2e; letter-spacing: -0.3px; }
.brand-sub  { font-size: 0.65rem; color: #aaa; letter-spacing: 0.4px; margin-top: 2px; }

.section-label {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #bbb;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    margin-top: 1.4rem;
}

.doc-card {
    background: #fff8f2;
    border: 1px solid #ffd4a8;
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 10px;
}

.doc-card-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1a1a2e;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 8px;
}

.doc-pills { display: flex; gap: 8px; }

.pill {
    background: #fff;
    border: 1px solid #ffd4a8;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.68rem;
    color: #ff6b00;
    font-weight: 500;
}

.badge-ready {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(255,107,0,0.08);
    color: #ff6b00;
    border: 1px solid rgba(255,107,0,0.2);
    margin-top: 10px;
}

.badge-waiting {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background: #f5f5f5;
    color: #bbb;
    border: 1px solid #e8e8e8;
    margin-top: 10px;
}

.model-info {
    font-size: 0.7rem;
    color: #ccc;
    line-height: 2;
}

.chat-header {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    max-width: 760px;
    margin: 0 auto;
}

.chat-header h1 {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}

.chat-header p { color: #bbb; font-size: 0.82rem; }

.chat-wrap {
    max-width: 760px;
    margin: 0 auto;
    padding: 1.2rem 1rem 7rem;
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.msg-row { display: flex; align-items: flex-end; gap: 10px; }
.msg-row-user { justify-content: flex-end; }
.msg-row-bot  { justify-content: flex-start; }

.avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
}

.av-bot  {
    background: linear-gradient(135deg, #ff6b00, #ff9a3c);
    box-shadow: 0 2px 8px rgba(255,107,0,0.25);
}

.av-user {
    background: linear-gradient(135deg, #1a1a2e, #2d2d4e);
    box-shadow: 0 2px 8px rgba(26,26,46,0.2);
}

.bubble {
    max-width: 66%;
    padding: 11px 15px;
    font-size: 0.855rem;
    line-height: 1.65;
    word-wrap: break-word;
}

.bubble-user {
    background: linear-gradient(135deg, #1a1a2e, #2d2d4e);
    color: #ffffff;
    border-radius: 16px 16px 3px 16px;
    box-shadow: 0 3px 12px rgba(26,26,46,0.18);
}

.bubble-bot {
    background: #ffffff;
    color: #1a1a2e;
    border: 1px solid #ffe0c2;
    border-radius: 16px 16px 16px 3px;
    box-shadow: 0 3px 10px rgba(255,107,0,0.07);
}

.msg-time { font-size: 0.62rem; color: #ccc; margin-top: 4px; padding: 0 3px; }
.time-user { text-align: right; }
.time-bot  { text-align: left; }

.empty-state { text-align: center; padding: 2.5rem 2rem 1rem; }
.empty-icon  { font-size: 2.5rem; opacity: 0.18; margin-bottom: 1rem; }
.empty-state h3 { font-size: 0.95rem; font-weight: 600; color: #bbb; margin-bottom: 0.4rem; }
.empty-state p  { font-size: 0.77rem; color: #ddd; }

.upload-prompt {
    max-width: 520px;
    margin: 1.5rem auto 0;
    padding: 0 1rem;
}

.upload-prompt-label {
    text-align: center;
    font-size: 0.8rem;
    color: #ff6b00;
    font-weight: 600;
    margin-bottom: 0.7rem;
}

.chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 1.4rem; }

.chip {
    background: #fff8f2;
    border: 1px solid #ffd4a8;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.72rem;
    color: #ff6b00;
    font-weight: 500;
}

[data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 1.5px solid #ffe0c2 !important;
    border-radius: 14px !important;
    color: #1a1a2e !important;
    padding: 13px 18px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.855rem !important;
    box-shadow: 0 2px 8px rgba(255,107,0,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #ff6b00 !important;
    box-shadow: 0 0 0 3px rgba(255,107,0,0.1) !important;
}

[data-testid="stTextInput"] input::placeholder { color: #ddd !important; }

[data-testid="stFileUploader"] {
    background: #fff8f2;
    border: 1.5px dashed #ffd4a8;
    border-radius: 12px;
    padding: 0.6rem;
}

[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; }

button[kind="primary"], [data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #ff6b00, #ff9a3c) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    box-shadow: 0 4px 14px rgba(255,107,0,0.3) !important;
    transition: opacity 0.2s, transform 0.1s !important;
}

button[kind="primary"]:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

button[kind="secondary"], [data-testid="baseButton-secondary"] {
    background: #fff8f2 !important;
    border: 1px solid #ffd4a8 !important;
    border-radius: 12px !important;
    color: #ff6b00 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    transition: background 0.2s !important;
}

button[kind="secondary"]:hover { background: #fff0e0 !important; }

.stSpinner > div { border-top-color: #ff6b00 !important; }

div[data-testid="stMarkdownContainer"] p { color: #1a1a2e; }

/* Streamlit branding chhupao, magar header ko zinda rakho */
footer, #MainMenu, [data-testid="stToolbar"] { display: none !important; }

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Sidebar band hone par expand button — ye hamesha nazar aana chahiye */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    top: 12px !important;
    left: 12px !important;
    background: #ffffff !important;
    border: 1.5px solid #ffd4a8 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    box-shadow: 0 2px 10px rgba(255,107,0,0.15) !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    color: #ff6b00 !important;
    fill: #ff6b00 !important;
}

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ffd4a8; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_embedding_fn():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_llm_client():
    return InferenceClient(api_key=HF_TOKEN)


def build_collection(chunks: list, pdf_name: str):
    embedding_fn = get_embedding_fn()
    client       = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("pdf_qa")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="pdf_qa",
        embedding_function=embedding_fn
    )
    ids       = [str(i) for i in range(len(chunks))]
    metadatas = [{"source": pdf_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return collection


def retrieve_context(collection, query: str, top_k: int = 5) -> str:
    results = collection.query(query_texts=[query], n_results=top_k)
    return "\n\n---\n\n".join(results["documents"][0])


def ask_llm(prompt: str) -> str:
    client = get_llm_client()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(10)
                continue
            return f"Error: {str(e)}"


def rag_answer(collection, question: str) -> str:
    context = retrieve_context(collection, question, top_k=5)
    prompt  = f"""You are a helpful assistant answering questions based on a document.
Use the context below to answer as completely as possible.
If related information exists in the context, use it to give a helpful answer.
Only say "I don't know based on the provided document" if the context is completely irrelevant.

Context:
{context}

Question: {question}

Answer:"""
    return ask_llm(prompt)


def process_pdf(uploaded_file) -> tuple:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader    = PyPDFLoader(tmp_path)
    pages     = loader.load()
    full_text = " ".join([p.page_content for p in pages])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(full_text)
    os.unlink(tmp_path)
    return chunks, len(pages)


def handle_upload(uploaded_file):
    """Sidebar aur main page — dono uploaders yahi function call karte hain."""
    if not uploaded_file:
        return
    if st.session_state.pdf_name == uploaded_file.name:
        return

    with st.spinner("Indexing document..."):
        chunks, num_pages = process_pdf(uploaded_file)
        st.session_state.collection    = build_collection(chunks, uploaded_file.name)
        st.session_state.pdf_name      = uploaded_file.name
        st.session_state.pdf_chunks    = len(chunks)
        st.session_state.pdf_pages     = num_pages
        st.session_state.messages      = []
        st.session_state.input_counter += 1
    st.rerun()


if "messages"      not in st.session_state: st.session_state.messages      = []
if "collection"    not in st.session_state: st.session_state.collection    = None
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = None
if "pdf_chunks"    not in st.session_state: st.session_state.pdf_chunks    = 0
if "pdf_pages"     not in st.session_state: st.session_state.pdf_pages     = 0
if "input_counter" not in st.session_state: st.session_state.input_counter = 0


with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-name">DocChat AI</div>
            <div class="brand-sub">RAG · LLaMA 3.1 · ChromaDB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

 st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    if st.session_state.collection:
        sidebar_file = st.file_uploader(
            "PDF Upload",
            type=["pdf"],
            label_visibility="collapsed",
            key="sidebar_uploader"
        )
        handle_upload(sidebar_file)

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-name">📄 {st.session_state.pdf_name}</div>
            <div class="doc-pills">
                <span class="pill">{st.session_state.pdf_pages} pages</span>
                <span class="pill">{st.session_state.pdf_chunks} chunks</span>
            </div>
        </div>
        <div class="badge-ready">● Ready to chat</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-waiting">○ No document loaded</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.input_counter += 1
        st.rerun()

    st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-info">
        🤖 &nbsp;LLaMA 3.1 · 8B Instruct<br>
        🔍 &nbsp;all-MiniLM-L6-v2<br>
        🗄️ &nbsp;ChromaDB · Persistent<br>
        📦 &nbsp;Top-K: 5 · Chunk: 500
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="chat-header">
    <h1>Chat with your Document</h1>
    <p>Answers grounded in your uploaded PDF</p>
</div>
""", unsafe_allow_html=True)


# Koi document load nahi hua -> main page par hi uploader dikhao.
# Isse sidebar band ho to bhi app kaam karti rehti hai.
if not st.session_state.collection:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>No conversation yet</h3>
        <p>Upload a PDF below, then ask your first question</p>
    </div>
    """, unsafe_allow_html=True)

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        main_file = st.file_uploader(
            "Upload a PDF to begin",
            type=["pdf"],
            key="main_uploader"
        )
        handle_upload(main_file)

    st.markdown("""
    <div class="chips">
        <span class="chip">What is this document about?</span>
        <span class="chip">Summarize the key points</span>
        <span class="chip">What are the main topics?</span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <h3>Document ready</h3>
            <p>Ask your first question below</p>
            <div class="chips">
                <span class="chip">What is this document about?</span>
                <span class="chip">Summarize the key points</span>
                <span class="chip">What are the main topics?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            t    = msg.get("time", "")
            text = msg["content"]

            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row msg-row-user">
                    <div>
                        <div class="bubble bubble-user">{text}</div>
                        <div class="msg-time time-user">{t}</div>
                    </div>
                    <div class="avatar av-user">👤</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-row msg-row-bot">
                    <div class="avatar av-bot">🧠</div>
                    <div>
                        <div class="bubble bubble-bot">{text}</div>
                        <div class="msg-time time-bot">{t}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([11, 1])
    with col1:
        user_input = st.text_input(
            "chat_input",
            placeholder="Ask a question about your document...",
            label_visibility="collapsed",
            key=f"input_{st.session_state.input_counter}"
        )
    with col2:
        send = st.button("➤", type="primary", use_container_width=True)

    if send and user_input.strip():
        question = user_input.strip()
        now      = datetime.now().strftime("%H:%M")

        st.session_state.messages.append({"role": "user", "content": question, "time": now})

        with st.spinner("Thinking..."):
            answer = rag_answer(st.session_state.collection, question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%H:%M")
        })

        st.session_state.input_counter += 1
        st.rerun()
