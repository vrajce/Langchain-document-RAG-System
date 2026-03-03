# 🤖 LangChain Document RAG System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange.svg)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-red.svg)](https://github.com/facebookresearch/faiss)

A high-performance **Retrieval-Augmented Generation (RAG)** system built with LangChain. This project allows you to "chat" with your local documents using state-of-the-art open-source LLMs and local vector embeddings.

---

## 🚀 Cool Features

- **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally—no API costs for vectorization! 💸
- **Brainy LLM**: Powered by `Mistral-7B-Instruct-v0.3` via Hugging Face Inference API for smart, human-like responses. 🧠
- **Ultra-Fast Search**: Uses **FAISS** (Facebook AI Similarity Search) for lightning-fast document retrieval. ⚡
- **Expert Mode**: Configured with a specialized LangChain Expert system prompt. 🎓
- **Source Tracking**: The bot doesn't just answer; it tells you exactly which document it used for the information! 📚

---

## 🛠️ Tech Stack

- **Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **Embeddings**: [HuggingFace (Local)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Environment**: [Dotenv](https://github.com/theskumar/python-dotenv) for secure API management

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/vrajce/Langchain-document-RAG-System.git
cd Langchain-document-RAG-System
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets
Create a `.env` file in the root directory:
```env
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

---

## 📖 How to Use

### Phase 1: Ingest Documents
Place your `.mdx` or text documents in the `/docs` folder, then run:
```bash
python ingest.py
```
*This will chunk your data and save a local "brain" in the `faiss_index_react/` folder.*

### Phase 2: Start the Chat Bot
Once ingestion is complete, start your specialized assistant:
```bash
python bot.py
```

---

## 🏗️ Project Structure

- `ingest.py`: Handles document loading, splitting, and vector storage creation.
- `bot.py`: The main chat interface with RetrievalQA chain.
- `insert.py`: A handy script for testing local embeddings.
- `docs/`: Your source documents go here!

---

## 🤝 Contributing
Feel free to open issues or submit pull requests! Let's make this RAG system even cooler! 🔥
