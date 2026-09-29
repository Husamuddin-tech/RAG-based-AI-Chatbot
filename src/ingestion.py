import argparse
import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


# Keep comfortably below the current free-tier embedding request limit.
BATCH_SIZE = 50

# Wait long enough for the minute-based quota window to reset.
BATCH_WAIT_SECONDS = 65


def setup_pinecone_index(index_name: str) -> None:
    """Create the Pinecone index if it does not already exist."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set in the environment.")

    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Pinecone index already exists: {index_name}")


def run_ingestion(pdf_path: str, index_name: str):
    """Load, chunk, embed, and store the Agentic AI eBook."""

    # ---------------------------------------------------------
    # 1. Load document
    # ---------------------------------------------------------
    print(f"Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages.")

    # ---------------------------------------------------------
    # 2. Chunk document
    # ---------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    # ---------------------------------------------------------
    # 3. Pinecone index
    # ---------------------------------------------------------
    setup_pinecone_index(index_name)

    # ---------------------------------------------------------
    # 4. Gemini embeddings
    # ---------------------------------------------------------
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=1536,
    )

    # ---------------------------------------------------------
    # 5. Create vector store
    # ---------------------------------------------------------
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    # ---------------------------------------------------------
    # 6. Upload embeddings in controlled batches
    # ---------------------------------------------------------
    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_chunks)
        batch = chunks[start:end]

        batch_number = (start // BATCH_SIZE) + 1
        total_batches = (
            (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
        )

        print(
            f"Embedding batch {batch_number}/{total_batches} "
            f"({start + 1}-{end} of {total_chunks})..."
        )

        vector_store.add_documents(batch)

        print(
            f"Uploaded batch {batch_number}/{total_batches}."
        )

        # Wait between batches except after the final batch.
        if end < total_chunks:
            print(
                f"Waiting {BATCH_WAIT_SECONDS} seconds "
                "before the next embedding batch..."
            )
            time.sleep(BATCH_WAIT_SECONDS)

    print(
        f"Ingestion completed successfully. "
        f"Stored {total_chunks} chunks in Pinecone."
    )

    return vector_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the Agentic AI eBook into Pinecone."
    )

    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="data/Ebook-Agentic-AI.pdf",
        help="Path to the Agentic AI eBook PDF.",
    )

    parser.add_argument(
        "index_name",
        nargs="?",
        default=os.getenv(
            "PINECONE_INDEX_NAME",
            "agentic-ai-index",
        ),
        help="Pinecone index name.",
    )

    args = parser.parse_args()

    run_ingestion(
        args.pdf_path,
        args.index_name,
    )


if __name__ == "__main__":
    main()