# RAG Agentic AI

RAG-based AI Chatbot using Python, LangGraph, Pinecone, OpenAI, and FastAPI.

The project uses the **Agentic AI eBook** as the knowledge source and follows the provided reference requirements.

## Requirements

- Python 3.10 or higher
- OpenAI API key
- Pinecone API key
- Pinecone index configured with dimension `1536` and cosine distance

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── graph.py
│   └── config.py
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── tests_sample_queries.py
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\\Scripts\\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `.env` from `.env.example` and set:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

### 4. Add the knowledge source

Place the Agentic AI eBook at:

```text
data/Ebook-Agentic-AI.pdf
```

### 5. Ingest the document

```bash
python -m src.ingestion data/Ebook-Agentic-AI.pdf agentic-ai-index
```

### 6. Run the FastAPI application

```bash
uvicorn app:app --reload
```

The API exposes:

```text
POST /chat
```

Request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Response contains:

- `final_answer`
- `retrieved_context`
- `confidence_score`

## RAG Workflow

```text
Question
   ↓
Retrieve top-k relevant chunks from Pinecone
   ↓
LangGraph state
   ↓
Generate answer using retrieved context only
   ↓
Return answer + context + confidence score
```

## Benchmark Queries

The project includes `tests_sample_queries.py` with the reference benchmark questions, including an out-of-context FIFA World Cup question used to validate grounding.
