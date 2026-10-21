
import streamlit as st
from pathlib import Path

from rag_Pipeline import build_vector_store, ask_question


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #dbeafe;
        margin-bottom: 0;
    }

    .source-note {
        color: #64748b;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load the existing local RAG pipeline once
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_store():
    return build_vector_store()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Your local document retrieval system")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        """
        1. Your documents are loaded.
        2. Text is converted into embeddings.
        3. ChromaDB searches for relevant chunks.
        4. The matching information is displayed with its source.
        """
    )

    st.divider()

    st.markdown("### Technology")
    st.markdown(
        """
        - Python
        - Streamlit
        - LangChain
        - Sentence Transformers
        - ChromaDB
        """
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Main interface
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Document Question Answering</h1>
        <p>Ask a question and discover relevant information
        from your documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This application retrieves relevant document passages. "
    "It does not generate new answers using an LLM."
)


# --------------------------------------------------
# Initialize session history
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Initialize the vector database
# --------------------------------------------------
try:
    with st.spinner(
        "Loading documents and preparing the local search engine..."
    ):
        vector_store = load_vector_store()

    st.success("Document search engine is ready!")

except Exception as error:
    st.error(f"Could not initialize the RAG pipeline: {error}")
    st.info(
        "Check that task2_rag_assistant/documents contains "
        "at least one readable TXT or PDF file."
    )
    st.stop()


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Accept a new question
# --------------------------------------------------
question = st.chat_input(
    "Ask something about your documents..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                response = ask_question(vector_store, question)

                st.markdown("### 🔎 Relevant information")
                st.markdown(response)

                st.caption(
                    "Information above was retrieved from your documents."
                )

            except Exception as error:
                response = f"Search failed: {error}"
                st.error(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
