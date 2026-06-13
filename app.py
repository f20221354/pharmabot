import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="PharmaBot",
    page_icon="💊",
    layout="wide"
)

# ================= SIDEBAR =================

st.sidebar.title("💊 PharmaBot")
st.sidebar.write("Ask questions about the pharmaceutical annual report.")

st.sidebar.divider()

st.sidebar.write("Tech Stack")
st.sidebar.caption(
    "Ollama Embeddings • ChromaDB • Gemini 2.5 Flash • Streamlit"
)

# ================= LOAD MODELS =================

@st.cache_resource
def load_components():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    return retriever, llm


retriever, llm = load_components()

# ================= SESSION STATE =================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= HEADER =================

st.title("💊 PharmaBot")
st.caption("Chat with your Annual Report")

# ================= DISPLAY HISTORY =================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg:

            with st.expander("📄 Sources"):

                for i, src in enumerate(msg["sources"], start=1):

                    st.write(
                        f"**Source {i} - Page {src['page']}**"
                    )

                    st.caption(src["text"][:300] + "...")

# ================= USER INPUT =================

question = st.chat_input(
    "Ask a question about the report..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Searching report..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = f"""
You are a pharmaceutical company assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, reply:

"I don't have that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        answer = response.content

        sources = []

        for doc in docs:

            sources.append(
                {
                    "page": doc.metadata.get(
                        "page",
                        "Unknown"
                    ),
                    "text": doc.page_content
                }
            )

    with st.chat_message("assistant"):

        st.write(answer)

        with st.expander("📄 Sources"):

            for i, src in enumerate(
                sources,
                start=1
            ):

                st.write(
                    f"**Source {i} - Page {src['page']}**"
                )

                st.caption(
                    src["text"][:300] + "..."
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )

# ================= FOOTER =================

st.divider()
st.caption(
    "🚀 Powered by ChromaDB + Ollama Embeddings + Gemini"
)