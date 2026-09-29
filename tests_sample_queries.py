# SAMPLE_QUERIES = [
#     "What is Agentic AI according to the eBook?",
#     "How do AI agents differ from traditional automation systems?",
#     "What are the core components of an Agentic Architecture?",
#     "What role does memory play in Agentic AI workflows?",
#     "Who won the 2022 FIFA World Cup?",
# ]



import os

from dotenv import load_dotenv

from src.graph import build_rag_graph


load_dotenv()


def run_query(graph, question: str) -> None:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    initial_state = {
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0,
    }

    result = graph.invoke(initial_state)

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED CONTEXT:")

    for i, chunk in enumerate(result["context"], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)

    print("\nCONFIDENCE SCORE:")
    print(result["score"])


def main() -> None:
    index_name = os.getenv(
        "PINECONE_INDEX_NAME",
        "agentic-ai-index",
    )

    graph = build_rag_graph(index_name)

    test_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]

    for question in test_queries:
        run_query(graph, question)


if __name__ == "__main__":
    main()