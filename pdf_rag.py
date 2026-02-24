"""
Minimal RAG (Retrieval-Augmented Generation) example with LangChain
Embeds multiple PDFs and answers questions about them
"""

import requests
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. DOWNLOAD PDFs FROM URLs
pdf_urls = {
    "transformer.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "yolo_v3.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "gpt3.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "bert.pdf": "https://arxiv.org/pdf/1907.11692.pdf",
    "electra.pdf": "https://arxiv.org/pdf/1910.10683.pdf",
}

all_documents = []

for pdf_name, pdf_url in pdf_urls.items():
    if not Path(pdf_name).exists():
        print(f"Downloading {pdf_name}...")
        response = requests.get(pdf_url)
        with open(pdf_name, "wb") as f:
            f.write(response.content)
        print(f"✓ {pdf_name} downloaded!")
    
    # 2. LOAD PDF
    print(f"Loading {pdf_name}...")
    loader = PyPDFLoader(pdf_name)
    documents = loader.load()
    all_documents.extend(documents)
    print(f"✓ {pdf_name} loaded ({len(documents)} pages)")

print(f"\nTotal documents loaded: {len(all_documents)} pages")

# 3. SPLIT DOCUMENTS INTO CHUNKS
print("Splitting documents...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(all_documents)
print(f"✓ Split into {len(docs)} chunks")

# 4. CREATE EMBEDDINGS & VECTOR STORE
print("Creating embeddings (this may take a minute)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)
print("✓ Vector store created!")

# 5. CREATE LLM
llm = OllamaLLM(model="mistral")

# 6. INTERACTIVE QUERY LOOP
print("\n" + "="*60)
print("RAG System Ready! Ask questions about the papers.")
print("Type 'q' to quit.")
print("="*60)

while True:
    print()
    query = input("Your Question (or 'q' to quit): ").strip()
    
    # Check if user wants to quit
    if query.lower() == 'q':
        print("Goodbye! 👋")
        break
    
    # Skip empty queries
    if not query:
        print("Please enter a question.")
        continue
    
    print("Searching documents...")
    
    # Search for relevant documents
    relevant_docs = vector_store.similarity_search(query, k=5)
    
    # Combine relevant documents into context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create a prompt with context
    prompt = f"""Based on the following context from the research papers, answer the question:

Context:
{context}

Question: {query}

Answer:"""
    
    # Get response from LLM
    print("\nGenerating response...\n")
    response = llm.invoke(prompt)
    print(f"Response:\n{response}\n")
