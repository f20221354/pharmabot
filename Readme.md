# PharmaBot: RAG-Powered Document Q&A

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about pharmaceutical company documents by searching relevant context and generating accurate answers using Google Gemini.

## What It Does

- 📄 Upload a PDF (annual report, drug handbook, etc.)
- 💬 Ask questions in natural language
- 🔍 Get answers backed by specific document excerpts
- 📚 See which document pages were used for each answer

## Tech Stack

- **LangChain** - LLM orchestration framework
- **Google Gemini** - LLM for answer generation
- **ChromaDB** - Vector database for semantic search
- **Streamlit** - Web interface
- **PyPDF** - PDF loading

## Setup

1. Clone this repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Create `.env` file with `GOOGLE_API_KEY=your_key_here`
5. Run `python ingest.py` to load your PDF
6. Run `streamlit run app.py`

## Live Demo

[Coming soon - Streamlit Cloud link]
