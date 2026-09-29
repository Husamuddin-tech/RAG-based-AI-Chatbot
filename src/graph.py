from typing import List, TypedDict

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph(index_name: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=1536,
    )

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
    )

    llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
)

    def retrieve_node(state: AgentState):
        docs = retriever.invoke(state["question"])

        context_texts = [
            doc.page_content
            for doc in docs
        ]

        return {
            "context": context_texts
        }

    def generate_node(state: AgentState):
        context_str = "\n\n".join(state["context"])

        prompt = f"""
    You are a strict document-grounded assistant.

    Answer the user's question using ONLY the retrieved context below.

    Rules:
    1. Use only information contained in the retrieved context.
    2. Do not use outside knowledge.
    3. The context may express the answer using different wording than the question.
    4. You may combine relevant information from multiple retrieved chunks.
    5. If the retrieved context contains enough information to answer, answer the question directly.
    6. Refuse ONLY when the retrieved context does not contain enough information.
    7. When the context is insufficient, state exactly:
    I cannot answer based on the provided document.

    Retrieved Context:
    {context_str}

    Question:
    {state["question"]}
    """

        response = llm.invoke(prompt)

        # Gemini may return structured content instead of a plain string.
        if isinstance(response.content, str):
            answer = response.content
        elif isinstance(response.content, list):
            text_parts = []

            for item in response.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)

            answer = "\n".join(
                part for part in text_parts if part
            ).strip()
        else:
            answer = str(response.content)

        # Confidence heuristic required by the reference implementation.
        confidence = 0.95 if len(state["context"]) > 0 else 0.0

        return {
            "answer": answer,
            "score": confidence,
        }

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "retrieve",
        retrieve_node,
    )

    workflow.add_node(
        "generate",
        generate_node,
    )

    workflow.add_edge(
        START,
        "retrieve",
    )

    workflow.add_edge(
        "retrieve",
        "generate",
    )

    workflow.add_edge(
        "generate",
        END,
    )

    return workflow.compile()