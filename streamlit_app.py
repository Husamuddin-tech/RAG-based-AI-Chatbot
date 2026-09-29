import json
import urllib.error
import urllib.request

import streamlit as st


FASTAPI_URL = "http://127.0.0.1:8000/chat"


st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)


def call_chat_api(query: str) -> dict:
    """Send the user query to the FastAPI /chat endpoint."""
    payload = json.dumps({"query": query}).encode("utf-8")

    request = urllib.request.Request(
        FASTAPI_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)

    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8")

        try:
            details = json.loads(error_body)
        except json.JSONDecodeError:
            details = error_body

        raise RuntimeError(
            f"FastAPI returned HTTP {error.code}: {details}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Unable to connect to the FastAPI server. "
            "Make sure the API is running on http://127.0.0.1:8000."
        ) from error


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []

if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = 0.0


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🤖 Agentic AI RAG Chatbot")

st.write(
    "Ask questions based on the provided Agentic AI eBook."
)


# ---------------------------------------------------------
# Display conversation
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Retrieved Context")

    confidence = st.session_state.confidence_score

    st.metric(
        label="Confidence Score",
        value=f"{confidence:.2f}",
    )

    st.divider()

    context = st.session_state.retrieved_context

    if context:
        for index, chunk in enumerate(context, start=1):
            with st.expander(f"Context Chunk {index}"):
                st.text(chunk)
    else:
        st.info(
            "Retrieved context will appear here after "
            "you submit a question."
        )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

query = st.chat_input(
    "Ask a question about the Agentic AI eBook..."
)


if query:
    # Display user message immediately
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Call FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                result = call_chat_api(query)

                answer = result["final_answer"]
                retrieved_context = result["retrieved_context"]
                confidence_score = result["confidence_score"]

                st.markdown(answer)

                # Store result for sidebar
                st.session_state.retrieved_context = (
                    retrieved_context
                )

                st.session_state.confidence_score = (
                    confidence_score
                )

                # Store assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # Refresh sidebar immediately
                st.rerun()

            except RuntimeError as error:
                error_message = str(error)

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )