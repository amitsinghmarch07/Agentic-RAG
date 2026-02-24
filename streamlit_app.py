"""
Streamlit Web UI for RAG System
Interactive Q&A interface for multiple PDFs
"""

import streamlit as st
import requests
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Page configuration
st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("📚 Agentic RAG - Research Papers Q&A")
st.markdown("Ask questions about multiple research papers and get instant answers!")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False

# PDFs to load
pdf_urls = {
    "Attention Is All You Need (Transformer)": "https://arxiv.org/pdf/1706.03762.pdf",
    "YOLOv3": "https://arxiv.org/pdf/1810.04805.pdf",
    "Language Models are Unsupervised Multitask Learners (GPT-3)": "https://arxiv.org/pdf/2005.14165.pdf",
    "BERT": "https://arxiv.org/pdf/1907.11692.pdf",
    "ELECTRA": "https://arxiv.org/pdf/1910.10683.pdf",
}

# AUTO-LOAD PDFs ON STARTUP
if not st.session_state.pdfs_loaded:
    with st.spinner("🔄 Loading and indexing PDFs... This may take 2-3 minutes"):
        all_documents = []
        
        # Download and load PDFs
        for pdf_name, pdf_url in pdf_urls.items():
            pdf_file = pdf_name.replace(" ", "_") + ".pdf"
            
            if not Path(pdf_file).exists():
                try:
                    response = requests.get(pdf_url, timeout=30)
                    with open(pdf_file, "wb") as f:
                        f.write(response.content)
                except Exception as e:
                    st.error(f"Error downloading {pdf_name}: {e}")
                    continue
            
            try:
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                st.error(f"Error loading {pdf_name}: {e}")
        
        if all_documents:
            # Split documents
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(all_documents)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
            st.session_state.pdfs_loaded = True
            st.success("✓ PDFs ready! You can now ask questions.")
        else:
            st.error("No PDFs were loaded. Please check your internet connection.")

# Sidebar - Info
with st.sidebar:
    st.header("ℹ️ Info")
    
    if st.session_state.pdfs_loaded:
        st.success("✓ PDFs Loaded and Ready")
        st.markdown("---")
        st.subheader("📚 Loaded Papers:")
        for paper in pdf_urls.keys():
            st.write(f"• {paper}")
    else:
        st.warning("Loading PDFs...")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📝 Chat History")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# Main content
if st.session_state.pdfs_loaded and st.session_state.vector_store is not None:
    # Question input
    query = st.text_input("🔍 Ask a question about the papers:")
    
    if query:
        # Initialize LLM
        llm = OllamaLLM(model="mistral")
        
        with st.spinner("Searching documents and generating response..."):
            # Search for relevant documents
            relevant_docs = st.session_state.vector_store.similarity_search(query, k=5)
            
            # Combine context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt
            prompt = f"""Based on the following context from the research papers, answer the question:

Context:
{context}

Question: {query}

Answer:"""
            
            # Get response
            response = llm.invoke(prompt)
        
        # Display response
        st.markdown("---")
        st.subheader("💡 Answer:")
        st.markdown(response)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": query,
            "answer": response
        })
        
        # Show relevant sources
        with st.expander("📄 Relevant Sources"):
            for i, doc in enumerate(relevant_docs, 1):
                st.markdown(f"**Source {i}:**")
                st.text(doc.page_content[:300] + "...")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📚 Previous Questions")
        for i, chat in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"Q{i}: {chat['question'][:50]}..."):
                st.write(chat['answer'])

else:
    st.info("� PDFs are loading... This may take 2-3 minutes on first run. Please refresh the page if it takes too long.")
