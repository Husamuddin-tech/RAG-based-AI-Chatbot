# RAG Agentic AI Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-2D2D2D?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-LLM%20%26%20Embeddings-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20Database-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

A Retrieval-Augmented Generation (RAG) AI chatbot that answers questions using the provided **Agentic AI eBook** as its knowledge source.

The application uses **Python, LangGraph, Pinecone, Google Gemini, FastAPI, and Streamlit**. The RAG workflow retrieves relevant chunks from the eBook and instructs the language model to answer **only from the retrieved document context**. When the required information is not available in the retrieved context, the chatbot refuses to answer with:

```text
I cannot answer based on the provided document.
```

> **Provider note:** The supplied reference specifies OpenAI for embeddings and generation. For this implementation, OpenAI was intentionally replaced with Google Gemini to meet the project's no-paid-AI requirement. The PDF ingestion, chunking, Pinecone, LangGraph, API, UI, output structure, and benchmark requirements remain aligned with the reference.

## Features

- PDF ingestion from the Agentic AI eBook.
- Recursive text splitting with `chunk_size=1000` and `chunk_overlap=200`.
- Gemini embeddings stored in Pinecone.
- Pinecone index using **1536 dimensions** and **cosine** similarity.
- LangGraph workflow: `START → retrieve → generate → END`.
- Top-5 relevant context chunks retrieved for each query.
- Strict document-grounded answer generation.
- Confidence score returned with every response.
- FastAPI `POST /chat` API.
- Swagger/OpenAPI documentation through FastAPI.
- Optional Streamlit chat interface with retrieved context and confidence score.
- Benchmark test script with 5 reference queries, including an out-of-context grounding test.

## Architecture

```text
                         Agentic AI eBook
                                │
                                ▼
                         PyPDFLoader
                                │
                                ▼
              RecursiveCharacterTextSplitter
                  chunk_size = 1000
                  chunk_overlap = 200
                                │
                                ▼
                    Gemini Embeddings
                     gemini-embedding-2
                                │
                                ▼
                           Pinecone
                    1536 dimensions / cosine
                                │
                                ▼
                         User Question
                                │
                                ▼
                           LangGraph
                                │
                         START → retrieve
                                │
                                ▼
                            generate
                                │
                                ▼
                              END
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
             FastAPI                         Streamlit
             /chat                             UI
                │                               │
                └───────────────┬───────────────┘
                                ▼
                  Answer + Context + Score
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| RAG framework | LangChain |
| Workflow orchestration | LangGraph |
| LLM | Google Gemini 3.5 Flash-Lite |
| Embeddings | Google Gemini Embedding 2 |
| Vector database | Pinecone |
| PDF processing | PyPDFLoader / pypdf |
| Text chunking | RecursiveCharacterTextSplitter |
| API | FastAPI |
| Server | Uvicorn |
| UI | Streamlit |
| Configuration | python-dotenv |

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── graph.py
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── tests_sample_queries.py
```

## Requirements

- Python 3.10 or higher.
- A Google Gemini API key.
- A Pinecone API key.
- The Agentic AI eBook PDF stored at `data/Ebook-Agentic-AI.pdf`.

The tested project environment uses Python 3.12.

## Installation

### 1. Clone or create the project

```bash
git clone <your-github-repository-url>
cd rag-agentic-ai
```

### 2. Create a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
```

A template is provided in `.env.example`.

### Security

Never commit the `.env` file or API keys to GitHub.

The repository should contain `.env.example`, not the real credentials.

## Pinecone Index

The project uses the following Pinecone configuration:

```text
Index name: agentic-ai-index
Dimension: 1536
Metric: cosine
Vector type: dense
```

The ingestion script creates the index when it does not already exist.

## Document Ingestion

Place the source document at:

```text
data/Ebook-Agentic-AI.pdf
```

Run:

```bash
python src/ingestion.py
```

The ingestion pipeline performs:

```text
PDF
 ↓
PyPDFLoader
 ↓
Text extraction
 ↓
RecursiveCharacterTextSplitter
 ↓
1000-character chunks / 200-character overlap
 ↓
Gemini embeddings
 ↓
Pinecone
```

During the tested run, the 60-page eBook produced **119 chunks**.

### Free-tier embedding batching

Gemini free-tier embedding requests can be rate-limited. The tested ingestion implementation therefore uploads chunks in controlled batches and waits between batches to avoid exceeding the observed per-minute embedding request limit.

The successful tested run processed:

```text
Batch 1: 50 chunks
Batch 2: 50 chunks
Batch 3: 19 chunks
```

and stored all **119 vectors** in Pinecone.

## Run the FastAPI Backend

Start the API from the project root:

```bash
uvicorn app:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

## API Endpoint

### `POST /chat`

Request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Response:

```json
{
  "final_answer": "According to the provided document, Agentic AI ...",
  "retrieved_context": [
    "...",
    "...",
    "..."
  ],
  "confidence_score": 0.95
}
```

The endpoint returns:

- `final_answer` — generated answer grounded in retrieved context.
- `retrieved_context` — retrieved document chunks.
- `confidence_score` — confidence value produced by the reference implementation's heuristic.

## Example API Test

Using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Agentic AI according to the eBook?"}'
```

## Streamlit UI

The reference describes Streamlit as an optional alternative interface. This project includes a Streamlit UI in addition to the FastAPI backend.

Start FastAPI first:

```bash
uvicorn app:app --reload
```

Then, in another terminal:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

The UI provides:

- Chat input for questions.
- Generated answers.
- Retrieved context chunks in the sidebar.
- Confidence score in the sidebar.

## LangGraph Workflow

The RAG graph uses four state fields:

```python
question
context
answer
score
```

Workflow:

```text
START
  ↓
retrieve
  ↓
generate
  ↓
END
```

### Retrieve node

The retriever queries Pinecone for the top relevant document chunks.

### Generate node

The Gemini model receives only the retrieved context and the user question. It is instructed not to use outside information.

If the retrieved context is insufficient, the response is:

```text
I cannot answer based on the provided document.
```

## Testing

Run the benchmark script:

```bash
python tests_sample_queries.py
```

The test suite uses the reference benchmark questions:

```text
1. What is Agentic AI according to the eBook?
2. How do AI agents differ from traditional automation systems?
3. What are the core components of an Agentic Architecture?
4. What role does memory play in Agentic AI workflows?
5. Who won the 2022 FIFA World Cup?
```

The first four questions should be answered from retrieved eBook context. The FIFA question is an out-of-context validation test and should return:

```text
I cannot answer based on the provided document.
```

### Tested Results

The complete 5-query benchmark was successfully executed after tuning retrieval to top-5 chunks. The grounded questions produced document-based answers, and the FIFA query was correctly refused.

## Grounding Behavior

The chatbot is designed to remain grounded in the retrieved eBook context.

```text
Question
   ↓
Pinecone retrieval
   ↓
Retrieved context
   ↓
Gemini generation
   ↓
Is sufficient information present?
   ├── Yes → Answer from context
   └── No  → Refuse using the required message
```

The system does not intentionally use external web search or a separate knowledge source for answering user questions.

## Reference Alignment

The implementation follows the supplied project reference for:

- RAG-based document question answering.
- Agentic AI eBook as the knowledge source.
- PDF ingestion and chunking.
- Pinecone vector storage.
- LangGraph retrieval/generation workflow.
- Strict grounding.
- Answer, retrieved context, and confidence score output.
- FastAPI `/chat` interface.
- Streamlit as an optional UI.
- 5–6 benchmark-style grounding tests.
- Clean Python project structure and README documentation.

The only intentional provider substitution is:

```text
Reference: OpenAI embeddings + OpenAI LLM
Implementation: Gemini embeddings + Gemini LLM
```

This change was made specifically to avoid paid OpenAI API usage.

## Submission Checklist

- [x] Working Python RAG implementation
- [x] Agentic AI eBook knowledge source
- [x] Document ingestion
- [x] Vector embeddings
- [x] Pinecone vector index
- [x] LangGraph workflow
- [x] Strict document grounding
- [x] FastAPI API output
- [x] Streamlit UI output
- [x] Retrieved context returned
- [x] Confidence score returned
- [x] 5 benchmark queries
- [x] Out-of-context refusal test
- [x] Setup and execution instructions
- [x] Architecture documentation

## Important Notes

- Keep API credentials in `.env` only.
- Keep `data/Ebook-Agentic-AI.pdf` available before running ingestion.
- The Pinecone index must remain compatible with the 1536-dimensional embeddings used by this implementation.
- Start FastAPI before using the Streamlit UI.
- Run document ingestion before testing retrieval against a newly created Pinecone index.
