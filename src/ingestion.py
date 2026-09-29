import argparse
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# Load environment variables from .env
load_dotenv()


def setup_pinecone_index(index_name: str) -> None:
    """Create the Pinecone index if it does not already exist."""
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is not set in the environment.")

    pc = Pinecone(api_key=api_key)

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
    """Load the PDF, split it into chunks, embed the chunks, and store them in Pinecone."""

    # 1. Load document
    print(f"Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages.")

    # 2. Chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    # 3. Create Pinecone index
    setup_pinecone_index(index_name)

    # 4. Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 5. Store embeddings in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )

    print(f"Stored {len(chunks)} chunks in Pinecone.")

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

    print(
        f"Ingestion completed for "
        f"'{args.pdf_path}' into Pinecone index "
        f"'{args.index_name}'."
    )


if __name__ == "__main__":
    main()