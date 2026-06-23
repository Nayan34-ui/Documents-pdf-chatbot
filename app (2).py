
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import os

st.set_page_config(page_title="DocMind - PDF Chatbot", page_icon="📄", layout="wide")

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #6c8cff;
    text-align: center;
    padding: 1rem 0;
}
.source-box {
    background: #1e2535;
    border: 1px solid #2a3044;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">◈ DocMind — PDF Chatbot</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8891aa'>Upload PDFs and ask questions — with source attribution</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Upload PDFs")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    process_btn = st.button("Process PDFs", type="primary", use_container_width=True)

    if process_btn and uploaded_files and groq_api_key:
        with st.spinner("Processing PDFs..."):
            docs = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            for pdf_file in uploaded_files:
                reader = PdfReader(pdf_file)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        splits = splitter.split_text(text)
                        for split in splits:
                            docs.append(Document(
                                page_content=split,
                                metadata={"page": page_num, "filename": pdf_file.name}
                            ))

            if not docs:
                st.error("No text found in PDFs.")
            else:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                    memory=memory,
                    return_source_documents=True,
                    output_key="answer"
                )
                st.session_state.chain = chain
                st.session_state.messages = []
                st.success(f"✅ {len(docs)} chunks indexed from {len(uploaded_files)} PDF(s)!")

if "chain" not in st.session_state:
    st.info("👈 Upload PDFs and enter your Groq API key in the sidebar to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Sources ({len(msg['sources'])} found)"):
                    for src in msg["sources"]:
                        st.markdown(f"""<div class="source-box">
                            <b>📄 {src['filename']} — Page {src['page']}</b><br>
                            <i>{src['excerpt']}</i>
                        </div>""", unsafe_allow_html=True)

    if question := st.chat_input("Ask a question about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain({"question": question})
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                sources = []
                seen = set()
                for doc in source_docs:
                    key = (doc.metadata["filename"], doc.metadata["page"])
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "filename": doc.metadata["filename"],
                            "page": doc.metadata["page"],
                            "excerpt": doc.page_content[:250] + "..."
                        })
st.markdown(answer)

if sources:
    with st.expander(f"📄 Sources ({len(sources)} found)"):
        for src in sources:
            st.markdown(
                f"""
                <div class="source-box">
                    <b>📄 {src['filename']} — Page {src['page']}</b><br>
                    <i>{src['excerpt']}</i>
                </div>
                """,
                unsafe_allow_html=True,
            )
