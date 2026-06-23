# DocMind — PDF AI Chatbot

A RAG-based chatbot that answers questions from PDF documents with source attribution.

## Live Demo
Live App: https://your-app.streamlit.app

## Architecture

User uploads PDF
      |
PyPDF2 extracts text (page by page)
      |
LangChain splits into chunks (500 chars, 100 overlap)
      |
HuggingFace Embeddings (all-MiniLM-L6-v2) → FAISS Vector Store
      |
User asks question → semantic search → top 5 chunks retrieved
      |
Groq LLM (Llama 3.3 70B) generates answer with sources
      |
Answer + Page numbers + Excerpts shown to user

## Features
- Upload multiple PDFs (up to 50MB)
- Semantic search using FAISS
- Source attribution with page numbers and excerpts
- Conversation memory (chat history maintained)
- Fast responses via Groq LLM

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| PDF Parsing | PyPDF2 |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| LLM | Groq - Llama 3.3 70B |

## Design Decisions

### Chunking Strategy
- Chunk size: 500 characters with 100 character overlap
- Overlap ensures context is not lost at chunk boundaries
- Page metadata preserved per chunk for source attribution

### Embedding Model
- Used all-MiniLM-L6-v2 — fast, free, no API key needed
- 384-dimensional vectors, strong semantic similarity performance

### Retrieval Approach
- FAISS vector store with cosine similarity
- Top 5 chunks retrieved per query
- Source deduplication to avoid showing same page twice

### Prompt Design
- ConversationalRetrievalChain maintains chat history
- Grounded answers from retrieved context only

## Setup Instructions

1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/docmind-pdf-chatbot
   cd docmind-pdf-chatbot

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

4. Get your free Groq API key
   https://console.groq.com

## Deployment
Deployed on Streamlit Cloud — connect GitHub repo and deploy in 2 minutes.
