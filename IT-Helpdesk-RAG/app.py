from pathlib import Path

import streamlit as st

from rag_engine import build_index, retrieve, generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IT Assist AI | Intelligent IT Support",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# PROFESSIONAL UI STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F5F7FB;
    }

    [data-testid="stSidebar"] {
        background-color: #101827;
    }

    [data-testid="stSidebar"] * {
        color: #E5EAF2;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #1A2638;
        border-radius: 12px;
        padding: 10px;
    }

    .hero {
        background: linear-gradient(
            120deg, #101827 0%, #1C3552 70%, #126E75 100%
        );
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        border: 1px solid #28465E;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.7px;
    }

    .hero p {
        color: #D4E2F1;
        font-size: 15px;
        margin: 0;
        line-height: 1.7;
    }

    .eyebrow {
        color: #5EEAD4;
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .section-heading {
        color: #17243A;
        font-size: 20px;
        font-weight: 700;
        margin: 12px 0 5px 0;
    }

    .muted {
        color: #64748B;
        font-size: 13px;
        line-height: 1.6;
    }

    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E1E8F0;
        border-radius: 14px;
        padding: 18px;
        min-height: 125px;
        margin-bottom: 8px;
    }

    .feature-card h4 {
        color: #17243A;
        margin: 0 0 8px 0;
        font-size: 15px;
    }

    .feature-card p {
        color: #64748B;
        font-size: 13px;
        line-height: 1.6;
        margin: 0;
    }

    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 40px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E1E8F0;
        padding: 15px;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #8491A5;
        font-size: 12px;
        padding: 25px 0 12px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "index": None,
    "chunks": [],
    "messages": [],
    "pending_question": "",
    "indexed_files": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_local_documents():
    """Read supported documents from the documents folder."""
    files = []

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}:
            files.append((path.name, path.read_bytes()))

    return files


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending_question = ""


def queue_question(question):
    st.session_state.pending_question = question


def show_sources(sources):
    """Display the passages retrieved for an answer."""
    if not sources:
        st.caption("No source passages were returned.")
        return

    with st.expander(
        f"📚 View supporting sources ({len(sources)})",
        expanded=False,
    ):
        for number, source in enumerate(sources, start=1):
            filename = source.get("source", "Unknown document")
            page = source.get("page")
            score = source.get("score")
            passage = source.get("text", "")

            if page is None:
                location = "Text document"
            else:
                location = f"Page {page}"

            st.markdown(f"**{number}. {filename}**")
            st.caption(f"{location}")

            if score is not None:
                st.caption(f"Retrieval score: {score:.3f}")

            st.write(passage)

            if number < len(sources):
                st.divider()


def process_question(question):
    """Retrieve evidence, generate an answer, and save chat history."""
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    results = []
    try:
        results = retrieve(
            question,
            st.session_state.index,
            st.session_state.chunks,
        )
        answer = generate_answer(question, results)

    except Exception:
        answer = (
            "I couldn't complete this request because an error occurred "
            "while searching the knowledge base. Please try again. "
            "If the issue continues, check the application terminal."
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results,
        }
    )


# --------------------------------------------------
# SIDEBAR: KNOWLEDGE BASE
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 20px 0;">
            <div style="font-size: 12px; letter-spacing: 2px;
                        color: #5EEAD4; font-weight: 700;">
                INTELLIGENT SUPPORT
            </div>
            <div style="font-size: 25px; font-weight: 800;
                        color: #FFFFFF; margin-top: 5px;">
                IT Assist AI
            </div>
            <div style="font-size: 12px; color: #AAB8CA;">
                Knowledge-powered IT help desk
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 📚 Knowledge Base")
    st.caption(
        "Use approved IT troubleshooting documents to provide "
        "evidence-based answers."
    )

    uploaded_files = st.file_uploader(
        "Add PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Only upload documents you are authorized to use.",
    )

    local_files = load_local_documents()

    st.caption(
        f"{len(local_files)} local document(s) found in the documents folder."
    )

    build_clicked = st.button(
        "🔄 Build / Refresh Index",
        type="primary",
        use_container_width=True,
    )

    if build_clicked:
        files_by_name = {
            name: content for name, content in local_files
        }

        # Uploaded files are added for this indexing session.
        if uploaded_files:
            for uploaded_file in uploaded_files:
                files_by_name[uploaded_file.name] = (
                    uploaded_file.getvalue()
                )

        files_to_index = list(files_by_name.items())

        if not files_to_index:
            st.error(
                "No documents found. Add a TXT or PDF file first."
            )
        else:
            try:
                with st.spinner(
                    "Reading documents and building the search index..."
                ):
                    index, chunks = build_index(files_to_index)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.indexed_files = [
                    name for name, _ in files_to_index
                ]
                st.session_state.messages = []

                st.success(
                    f"Knowledge base ready: {len(chunks)} chunks indexed."
                )

            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()

    if st.session_state.index is not None:
        st.markdown("### 🟢 System Status")
        st.success("Knowledge base is ready.")

        st.metric(
            "Indexed text chunks",
            len(st.session_state.chunks),
        )

        with st.expander("Indexed documents"):
            for filename in st.session_state.indexed_files:
                st.write(f"📄 {filename}")

    else:
        st.markdown("### ⚪ System Status")
        st.info("Build the knowledge base to start chatting.")

    st.divider()

    st.markdown("### 💬 Conversation")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat()
        st.rerun()

    st.divider()

    st.caption(
        "Security reminder: never upload passwords, API keys, "
        "personal secrets, or confidential documents without approval."
    )


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-POWERED IT SERVICE DESK</div>
        <h1>Your Intelligent IT Support Assistant</h1>
        <p>
            Find troubleshooting guidance from your IT knowledge base.
            Ask a question, review the answer, and inspect its supporting
            document passages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Knowledge Base",
        "Ready" if st.session_state.index is not None else "Not built",
    )

with col2:
    st.metric(
        "Text Chunks",
        len(st.session_state.chunks),
    )

with col3:
    total_questions = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "user"
    )
    st.metric("Questions in this session", total_questions)


# --------------------------------------------------
# SAFETY NOTICE
# --------------------------------------------------

st.warning(
    "Prototype notice: Answers may be incomplete or incorrect. "
    "Verify important instructions with authorized IT support. "
    "Do not enter passwords, access tokens, or confidential information."
)


# --------------------------------------------------
# WELCOME SCREEN AND SUGGESTED QUESTIONS
# --------------------------------------------------

if not st.session_state.messages:
    st.markdown(
        '<div class="section-heading">How can we help you today?</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Choose a common IT issue or type your own question below.</p>',
        unsafe_allow_html=True,
    )

    suggestions = [
        (
            "📶 Wi-Fi connection",
            "My Wi-Fi is not connecting. What should I check?",
        ),
        (
            "🔐 Password reset",
            "I forgot my password. What is the approved way to reset it?",
        ),
        (
            "🌐 VPN troubleshooting",
            "What information should I record if my VPN fails?",
        ),
        (
            "🖨️ Printer issue",
            "My printer is not printing. What should I check?",
        ),
    ]

    suggestion_columns = st.columns(2)

    for i, (label, question_text) in enumerate(suggestions):
        with suggestion_columns[i % 2]:
            st.button(
                label,
                key=f"suggestion_{i}",
                use_container_width=True,
                on_click=queue_question,
                args=(question_text,),
            )

    st.markdown("<br>", unsafe_allow_html=True)

    feature_columns = st.columns(3)

    features = [
        (
            "🔎 Document retrieval",
            "Search your indexed troubleshooting documents for relevant guidance.",
        ),
        (
            "🧠 AI-generated answers",
            "Use your existing RAG and Ollama pipeline to formulate responses.",
        ),
        (
            "📖 Source transparency",
            "Inspect the passages retrieved to support each response.",
        ),
    ]

    for column, (title, description) in zip(feature_columns, features):
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            show_sources(message["sources"])


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

typed_question = st.chat_input(
    "Describe your IT issue or ask a troubleshooting question..."
)

pending_question = st.session_state.pending_question
st.session_state.pending_question = ""

question = typed_question or pending_question

if question:
    if st.session_state.index is None:
        st.warning(
            "Please build the knowledge base from the sidebar before asking questions."
        )
    else:
        process_question(question)
        st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        IT Assist AI · Retrieval-Augmented Generation (RAG) Prototype<br>
        Built with Streamlit, FAISS, Sentence Transformers and Ollama
    </div>
    """,
    unsafe_allow_html=True,
)
