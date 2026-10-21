
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"


def build_vector_store():
    """Load documents, split them into chunks, and store their embeddings."""

    document_files = list(DOCUMENTS_DIR.glob("*.txt"))

    if not document_files:
        raise FileNotFoundError(
            f"No .txt documents found in {DOCUMENTS_DIR}"
        )

    documents = []

    for file_path in document_files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

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

    print(f"Loaded {len(documents)} document(s).")
    print(f"Created {len(chunks)} text chunk(s).")
    print("Vector database created successfully.")

    return vector_store


def ask_question(vector_store, question):
    """Retrieve document chunks relevant to a question."""

    results = vector_store.similarity_search(question, k=3)

    if not results:
        return "No relevant information found."

    return "\n\n".join(
        f"Source: {Path(doc.metadata.get('source', 'Unknown')).name}\n"
        f"{doc.page_content}"
        for doc in results
    )


if __name__ == "__main__":
    store = build_vector_store()

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\nRelevant information:")
        print(ask_question(store, question))
