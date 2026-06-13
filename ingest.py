"""
ingest.py: Load PDF → Split → Embed → Store in ChromaDB
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma  # <-- FIXED: Using updated integration package

# Load API key from .env
load_dotenv()
print("API Key Found:", bool(os.getenv("GOOGLE_API_KEY")))

# ============== STEP 1: LOAD PDF ==============
print("📄 Loading PDF...")
pdf_path = "Pharma_doc.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✓ Loaded {len(documents)} pages")

# ============== STEP 2: SPLIT INTO CHUNKS ==============
print("\n✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Each chunk = 1000 characters
    chunk_overlap=100     # 100 chars overallap
)
chunks = splitter.split_documents(documents)
# chunks = chunks[:50]

print(f"✓ Created {len(chunks)} chunks")
print(f"   First chunk preview: {chunks[0].page_content[:100].strip()}...")

# ============== STEP 3: CREATE EMBEDDINGS ==============
# print("\n🧠 Converting to embeddings...")
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="text-embedding-004"
# )
# print("✓ Embeddings model loaded")
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     client_options={"api_endpoint": "generativelanguage.googleapis.com"},
# )
# ============== STEP 3: CREATE EMBEDDINGS ==============
# print("\n🧠 Converting to embeddings...")
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="text-embedding-004"
# # )
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

batch_size = 50

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1}")
    vector_db.add_documents(batch)

print("Done!")
# # ============== STEP 4: STORE IN CHROMADB ==============
# print("\n💾 Storing in ChromaDB...")
# vector_db = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory="./chroma_db"
# )
# # vector_db.persist() <-- FIXED: Removed (Auto-persists in modern langchain-chroma)
# print("✓ Stored successfully!")
# print(f"   Database location: ./chroma_db")

# ============== TEST: SEARCH ==============
print("\n🔍 Test search:")
test_query = "main products"
results = vector_db.similarity_search(test_query, k=2)
print(f"   Query: '{test_query}'")
print(f"   Top result preview: {results[0].page_content[:150].strip()}...")

print("\n✅ ingest.py complete! ChromaDB is ready.")