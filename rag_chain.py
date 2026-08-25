from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load document
loader = TextLoader("notes.txt")
documents = loader.load()

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_rag_documents"
)

# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# LLM
llm = ChatOllama(model="llama3.2")

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
""")

# RAG chain
chain = (
    {
        "context": retriever,
        "question": lambda x: x
    }
    | prompt
    | llm
)

# Ask
question = input("Ask your document: ")

response = chain.invoke(question)

print("\nAI:", response.content)