import uuid
import asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from neo4j.rag_graphiti_backend import (
    agent,
    ingest_pdf,
    retrieve_all_threads
)

# =========================== Utilities ===========================

def generate_thread_id():
    return str(uuid.uuid4())  # ✅ UUID must be string-safe for persistence


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["pdf_ingested"] = False  # ✅ RESET ingestion flag
    add_thread(thread_id)


def load_conversation(thread_id):
    state = agent.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "pdf_ingested" not in st.session_state:
    st.session_state["pdf_ingested"] = False  # ✅ TRACK PDF INGESTION

# ======================= Async Event Loop FIX =====================

# ✅ Streamlit-safe persistent event loop
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state["event_loop"] = loop


def run_async(coro):
    """Safely run async code inside Streamlit"""
    loop = st.session_state["event_loop"]
    return loop.run_until_complete(coro)


add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================

st.sidebar.title("LangGraph PDF Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

# ✅ INGEST ONLY ONCE
if uploaded_pdf and not st.session_state["pdf_ingested"]:
    with st.sidebar.status("Indexing PDF…", expanded=True) as status:
        run_async(
            ingest_pdf(
                uploaded_pdf.getvalue(),
                filename=uploaded_pdf.name,
            )
        )

        st.session_state["pdf_ingested"] = True  # ✅ MARK AS INGESTED
        status.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.header("My Conversations")

for thread_id in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        history = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue

            history.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = history

# ============================ Main Layout ========================

st.title("Chatbot Using Graphiti")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document")

if user_input:
    # -------------------------------
    # Store USER message
    # -------------------------------
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status_box = {"box": None}

        async def ai_stream():
            final_response = ""

            async for state in agent.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="values",
            ):
                messages = state.get("messages", [])
                if not messages:
                    continue

                last = messages[-1]

                # -------------------------------
                # Tool handling
                # -------------------------------
                if isinstance(last, ToolMessage):
                    tool_name = getattr(last, "name", "tool")

                    if status_box["box"] is None:
                        status_box["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_box["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                    continue

                # -------------------------------
                # AI response handling
                # -------------------------------
                if isinstance(last, AIMessage):
                    final_response = last.content
                    placeholder.markdown(final_response)

            return final_response

        # -------------------------------
        # Run async safely
        # -------------------------------
        ai_response = run_async(ai_stream())

        # -------------------------------
        # ✅ Properly close status box
        # -------------------------------
        if status_box["box"] is not None:
            status_box["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    # -------------------------------
    # Store ASSISTANT message
    # -------------------------------
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_response}
    )
