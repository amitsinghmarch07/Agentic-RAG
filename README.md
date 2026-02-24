# Agentic RAG - Research Papers Q&A System

A web-based AI application that allows users to ask questions about multiple research papers and get instant answers using LangChain, embeddings, and local LLMs.

## 📋 Features

- 📚 **Multi-PDF Support** - Load and index up to 5 research papers simultaneously
- 🔍 **Intelligent Search** - Semantic similarity search to find relevant content
- 🤖 **AI Powered** - Uses Ollama (local) or OpenAI for generating responses
- 💬 **Interactive Chat** - Ask questions and get instant answers with source references
- 🎨 **Beautiful Web UI** - Built with Streamlit for easy interaction
- 📄 **Source Attribution** - See which documents answered your question

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed (the script will start it automatically)

### Setup (One Command!)

1. **Extract the zip file** to your desired location

2. **Navigate to the folder** and run:
   ```bash
   bash run.sh
   ```

That's it! The script will:
- ✓ Create virtual environment (if needed)
- ✓ Install all dependencies
- ✓ Start Ollama server
- ✓ Launch the Streamlit app

The app will open automatically at `http://localhost:8501`

### Run the Application

**Simply execute (handles everything automatically):**
```bash
bash run.sh
```

Or:
```bash
./run.sh
```

The script automatically:
- ✓ Creates/activates virtual environment
- ✓ Installs all dependencies  
- ✓ Starts Ollama server
- ✓ Launches the Streamlit app at http://localhost:8501

## 📖 How to Use

1. **Load PDFs**
   - Click the blue "🔄 Load & Index PDFs" button in the left sidebar
   - Wait for the PDFs to download and process (2-3 minutes first time)
   - You'll see ✓ checkmarks when complete

2. **Ask Questions**
   - Type your question in the search box
   - Press Enter
   - Get instant answers with source documents

3. **View History**
   - All previous questions are saved
   - Expand any question to see the answer again
   - Click "🗑️ Clear Chat" to reset

## 📚 Included Papers

1. **Attention Is All You Need (Transformer)** - https://arxiv.org/pdf/1706.03762.pdf
2. **YOLOv3** - https://arxiv.org/pdf/1810.04805.pdf
3. **Language Models are Unsupervised Multitask Learners (GPT-3)** - https://arxiv.org/pdf/2005.14165.pdf
4. **BERT** - https://arxiv.org/pdf/1907.11692.pdf
5. **ELECTRA** - https://arxiv.org/pdf/1910.10683.pdf

You can modify the URLs in `streamlit_app.py` to use different papers.

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (mistral model)
- **Embeddings**: Hugging Face (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Framework**: LangChain

## 📁 Project Structure

```
Agentic-RAG/
├── streamlit_app.py      # Main web application
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── run.sh               # Quick startup script
└── README.md            # This file
```

## ⚙️ Configuration

Edit `streamlit_app.py` to:
- Change the LLM model (search for `OllamaLLM(model="mistral")`)
- Modify PDF URLs in the `pdf_urls` dictionary
- Adjust chunk size or search parameters

## 🐛 Troubleshooting

**"Ollama endpoint not found"**
- Make sure Ollama is running: `ollama serve`
- Pull the model: `ollama pull mistral`

**"Import error" for packages**
- Activate venv: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**PDFs not loading**
- Check internet connection
- Verify PDF URLs are accessible
- Try refreshing the page

## 📦 Dependencies

All required packages are in `requirements.txt`:
- langchain
- langchain-community
- streamlit
- faiss-cpu
- huggingface-hub
- ollama
- And more...

## 🔐 API Keys

Currently uses local Ollama (no API keys needed). To use OpenAI instead:
1. Get your API key from [openai.com](https://openai.com)
2. Add to `.env`: `OPENAI_API_KEY=your-key-here`
3. Modify `streamlit_app.py` to use ChatOpenAI instead of OllamaLLM

## 📝 License

This project is open source and available for use.

## 👨‍💻 Support

For issues or questions, check the GitHub repository:
https://github.com/amitsinghmarch07/Agentic-RAG

---

**Happy Querying!** 🚀
