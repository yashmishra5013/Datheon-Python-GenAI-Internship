
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"


def build_vector_store():
    """Load TXT/PDF documents and create a local ChromaDB vector store."""

    document_files = (
        list(DOCUMENTS_DIR.glob("*.txt"))
        + list(DOCUMENTS_DIR.glob("*.pdf"))
    )

    if not document_files:
        raise FileNotFoundError(
            f"No .txt or .pdf documents found in {DOCUMENTS_DIR}"
        )

    documents = []

    for file_path in document_files:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        documents.extend(loader.load())

    if not documents:
        raise ValueError("The documents contain no readable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    print(f"Loaded {len(documents)} document page(s) or text file(s).")
    print(f"Created {len(chunks)} text chunk(s).")
    print("ChromaDB vector database created successfully.")

    return vector_store


def ask_question(vector_store, question):
    """Retrieve the most relevant document chunks for a question."""

    question = question.strip()

    if not question:
        return "Please enter a question."

    results = vector_store.similarity_search(question, k=3)

    if not results:
        return "No relevant information found."

    return "\n\n".join(
        f"Source: {Path(doc.metadata.get('source', 'Unknown')).name}\n"
        f"{doc.page_content}"
        for doc in results
    )


def main():
    """Run the interactive document question-answering application."""

    try:
        vector_store = build_vector_store()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return

    print("\nDocument Q&A Assistant is ready!")
    print("Ask a question or type 'exit' to quit.")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            print("\nRelevant information:")
            print(ask_question(vector_store, question))
        except Exception as error:
            print(f"Could not retrieve information: {error}")


if __name__ == "__main__":
    main()
