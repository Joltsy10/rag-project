# Study Assistant: RAG-Powered Document Q&A

A production-grade Retrieval-Augmented Generation system for querying personal study notes and documents. Built to support exam preparation by enabling natural language search across multiple document sources.

## Motivation

Most students accumulate notes across PDFs, text files, and web resources. Finding specific information across all of them during exam prep is slow and frustrating. This system lets you ask questions in plain English and get grounded, cited answers from your own material.

## Architecture

```
Documents (PDF / TXT / URL)
        ↓
  Load & Parse
        ↓
  Recursive Chunking (chunk_size=500, overlap=50)
        ↓
  Embedding (all-MiniLM-L6-v2, 384 dimensions)
        ↓
  ChromaDB Vector Store (persistent)
        ↓
User Query -> Embed -> MMR Similarity Search (k=4)
        ↓
  Retrieved Chunks + Prompt V2
        ↓
  Llama 3.1 70B via Groq
        ↓
  Grounded Answer with Source Citations
```

## Stack

| Component | Technology |
|-----------|------------|
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector Database | ChromaDB (persistent, local) |
| LLM | Llama 3.1 70B via Groq API |
| RAG Framework | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |

## Evaluation

Systematic hyperparameter search across 12 configurations (k in {2, 4, 8}, chunk_size in {200, 300, 400, 500}) using a custom LLM-as-judge evaluation framework with four metrics:

- **Faithfulness:** is every claim in the answer grounded in retrieved context?
- **Answer Relevancy:** does the answer address the question asked?
- **Context Recall:** did retrieval find the chunks needed to answer?
- **Completeness:** did the system actually answer or deflect?

### Optimal Configuration: k=4, chunk_size=500

| Metric | Score |
|--------|-------|
| Faithfulness | 0.9222 |
| Answer Relevancy | 0.9778 |
| Context Recall | 0.7818 |
| Completeness | 0.7273 |

### Retrieval Tuning Results (8B model, fixed config)

| Technique | Completeness | Verdict |
|-----------|-------------|---------|
| Baseline | 0.6818 | Starting point |
| + Query Rewriting | 0.5545 | Dropped |
| + MMR Retrieval | 0.7273 | Kept |
| + Prompt V2 | 0.7727 | Best overall |
| + Hybrid Search (0.5/0.5) | 0.7273 | Dropped |
| + Hybrid Search (0.7/0.3) | 0.7273 | No improvement |

**Key findings:**

Larger chunks (500) consistently outperform smaller chunks across all k values, preserving semantic context better for these document types. k=4 balances retrieval coverage without adding noise. Query rewriting hurt completeness on factual Q&A since raw questions are already precise search queries for this domain. MMR improved completeness by reducing redundant chunk retrieval. Hybrid search added noise for this dataset and was rejected. LLM-as-judge evaluation requires a completeness metric to avoid inflated scores from "I don't know" responses being rated as faithful.

## Project Structure

```
rag-project/
├── src/
│   ├── document_loader.py      # PDF, TXT, URL loading + chunking
│   ├── embeddings.py           # HuggingFace embedding model
│   ├── vector_store.py         # ChromaDB operations (add, search, clear)
│   └── rag_chain.py            # LangChain RAG chains + prompt variants
├── api/
│   └── app.py                  # FastAPI backend
├── app/
│   └── streamlit_app.py        # Streamlit frontend
├── evaluation/
│   ├── evaluate.py             # LLM-as-judge evaluation harness
│   ├── test_set.json           # 11 manually curated test questions
│   └── results_*.json          # All experiment results
├── data/                       # User documents (gitignored)
├── .chroma/                    # ChromaDB persistent store (gitignored)
├── .env                        # API keys (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

**Prerequisites:** Python 3.10+, Groq API key (free at console.groq.com)

```bash
git clone https://github.com/yourusername/rag-project
cd rag-project

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

echo "GROQ_API_KEY=your_key_here" > .env

streamlit run app/streamlit_app.py
```

**Docker:**

```bash
docker-compose up --build
```

## Usage

1. Upload PDF or TXT files, or paste a URL in the sidebar
2. Wait for documents to be processed and indexed
3. Ask questions in plain English
4. Answers include source citations showing which document was used
5. Use "Clear Session" to start fresh with new documents

## Supported Document Types

| Type | Notes |
|------|-------|
| PDF | Text-based PDFs only; scanned PDFs require OCR preprocessing |
| TXT | Plain text files, UTF-8 encoded |
| URL | Web pages parsed and cleaned automatically |

## Known Limitations

- Scanned and handwritten PDFs are not supported without OCR preprocessing
- Context recall drops for very specific factual queries due to embedding model limitations
- Groq free tier has rate limits; sustained heavy usage may hit them
- Clearing session removes all indexed documents from ChromaDB

## What I Learned

- Proper RAG evaluation requires more than eyeballing answers. LLM-as-judge needs a completeness metric to avoid rewarding "I don't know" responses
- Chunk size affects answer quality more than retrieval depth for this document type
- Query rewriting helps for conversational queries but hurts precise factual Q&A
- PDF parsing quality directly impacts embedding quality and downstream retrieval
- MMR retrieval consistently improves completeness by reducing redundant chunk selection

## Roadmap

- [ ] Agentic extension: ReAct agent that decides between local retrieval and web search
- [ ] Web search integration via Tavily API
- [ ] OCR support for scanned PDFs and handwritten notes
- [ ] Conversation memory for follow-up questions
- [ ] Multi-user support with separate ChromaDB collections

## Related Projects

[Energy Consumption Forecasting](github link): LSTM time series forecasting with FastAPI and Docker
