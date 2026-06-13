"""
rag_chain.py
Load ChromaDB -> Retrieve relevant chunks -> Send context to Gemini
"""

from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================
# STEP 1: LOAD ENV VARIABLES
# ==========================
load_dotenv()

# ==========================
# STEP 2: LOAD CHROMADB
# ==========================
print("📚 Loading ChromaDB...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print("✓ ChromaDB loaded!")

# ==========================
# STEP 3: CREATE RETRIEVER
# ==========================
retriever = vector_db.as_retriever(
    search_kwargs={"k": 4}
)

print("✓ Retriever ready!")

# ==========================
# STEP 4: LOAD GEMINI
# ==========================
print("🤖 Loading Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

print("✓ Gemini loaded!")

# ==========================
# STEP 5: CHAT LOOP
# ==========================
print("\n" + "=" * 60)
print("💬 PharmaBot Ready")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    print("\n🔍 Searching documents...")

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a pharmaceutical company assistant.

Answer ONLY using the context provided below.

If the answer is not available in the context,
say:

"I don't have that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

    print("🤖 Generating answer...")

    response = llm.invoke(prompt)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.content)

    print("\n📄 Sources:")

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "Unknown")

        print(
            f"\n[{i}] Page {page}"
        )

        preview = doc.page_content[:200].replace("\n", " ")

        print(preview + "...")