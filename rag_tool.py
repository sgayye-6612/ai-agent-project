from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool


# ============================================================
# LOAD NOTES
# ============================================================

loader = TextLoader("notes.txt")

documents = loader.load()


# ============================================================
# SPLIT DOCUMENT
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_documents"
)


# ============================================================
# RETRIEVAL TOOL
# ============================================================

@tool
def search_notes(question: str) -> str:
    """Search Sahithi's notes for information about her goals, learning, skills, name, and personal notes."""

    results = vector_store.similarity_search(
        question,
        k=3
    )

    if not results:
        return "No relevant information found."

    return "\n\n".join(
        document.page_content
        for document in results
    )