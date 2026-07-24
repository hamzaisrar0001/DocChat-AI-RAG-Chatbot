# 🧠 DocChat AI — PDF Question & Answer Chatbot

> An intelligent RAG-powered chatbot that lets you have a conversation with any PDF document. Upload your document, ask questions in natural language, and get accurate answers grounded in the content.

---

##  Features

- 📄 **PDF Upload** — Upload any PDF directly from the UI
- 💬 **WhatsApp-style Chat** — Clean, intuitive chat interface
- 🔍 **Semantic Search** — Finds the most relevant content using vector similarity
- 🤖 **LLaMA 3.1 8B** — Powered by Meta's LLaMA via HuggingFace Inference API
- 🗄️ **Persistent Vector Store** — ChromaDB stores embeddings on disk
- ⚡ **Auto Retry** — Handles API rate limits gracefully
- 🌐 **Cloud Ready** — Deployed on Streamlit Cloud

---

##  Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│   ChromaDB      │  ◄── PDF chunks indexed with all-MiniLM-L6-v2
│  Vector Store   │
└────────┬────────┘
         │  Top-5 relevant chunks
         ▼
┌─────────────────┐
│  Prompt Builder │  ◄── Context + Question
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLaMA 3.1 8B   │  ◄── HuggingFace Inference API
│  (HuggingFace)  │
└────────┬────────┘
         │
         ▼
      Answer
```

---

##  Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Meta LLaMA 3.1 8B Instruct |
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (Persistent) |
| PDF Loader | LangChain PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| API | HuggingFace Inference API |

---

##  Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/hamzaisrar0001/DocChat-AI-RAG-Chatbot.git
cd DocChat-AI-RAG-Chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
Create a `.env` file in the root directory:
```
CHROMA_HUGGINGFACE_API_KEY=hf_your_token_here
```
Get your free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `main` branch → `app.py`
5. Go to **Settings → Secrets** and add:
```toml
CHROMA_HUGGINGFACE_API_KEY = "hf_your_token_here"
```
6. Click **Deploy**

---

##  Project Structure

```
DocChat-AI-RAG-Chatbot/
│
├── app.py               # Streamlit UI + RAG pipeline
├── main.ipynb           # Development notebook
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

##  RAG Pipeline Parameters

| Parameter | Value |
|---|---|
| Chunk Size | 500 tokens |
| Chunk Overlap | 50 tokens |
| Top-K Retrieval | 5 chunks |
| Max Output Tokens | 400 |
| Temperature | 0.3 |

---

##  Requirements

```
streamlit
langchain
langchain-community
langchain-text-splitters
chromadb
sentence-transformers
pypdf
python-dotenv
huggingface_hub
```

---

##  Security Note

Never commit your `.env` file or expose your HuggingFace API token publicly. Always use environment variables or Streamlit Secrets for sensitive credentials.

---

## What I Would Improve
- Add conversation memory so users can ask follow-up questions
- Support multiple PDFs simultaneously
- Add source citation with page numbers from the document
- Deploy on cloud for public access (HuggingFace Spaces or AWS)
- Add user authentication for private documents

---

## Tradeoffs
- ChromaDB is fast for local use but doesn't scale well for very large documents
- LLaMA runs locally which ensures privacy but requires good hardware
- Chose RAG over fine-tuning because it allows dynamic knowledge updates without retraining the model

---

## Time Taken
Approximately 2.5 hours for this assessment submission

---

## 👨‍💻 Author

**Muhammad Hamza Israr**
[![GitHub](https://img.shields.io/badge/GitHub-hamzaisrar0001-black?logo=github)](https://github.com/hamzaisrar0001)

---
