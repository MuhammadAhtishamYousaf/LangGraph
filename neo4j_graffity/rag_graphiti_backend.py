from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

from langchain.agents import create_agent

# agent = create_agent("openai:gpt-5", tools=tools)
load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  

model = ChatOpenAI(model='gpt-4o-mini')


graphiti = Graphiti(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"]
)

# -------------------
# 2. PDF retriever store (per thread)
# -------------------

async def ingest_pdf(file_bytes: bytes, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100,
        )
        chunks = splitter.split_documents(docs)

        # 3️⃣ Convert chunks → Graphiti episodes
        for i, doc in enumerate(chunks):
            await graphiti.add_episode(
                name=f"file_chunk_{i}",
                episode_body=doc.page_content,
                source=EpisodeType.text,
                source_description=f"{filename}",
                reference_time=datetime.now(timezone.utc),
            )

            print(f"✅ Stored chunk {i}")

        print("🎯 All PDF chunks stored in Graphiti")

        
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            # await graphiti.close()
            os.remove(temp_path)
            # os.remove(file_bytes)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")



def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


async def rag_tool(query: str) -> dict:
    """async Tool to retrieve information from neo4j always use this if the query is about the document. 
    """
    print(f"7️⃣ rag tool called {query}")
    # Graphiti search (sync wrapper assumed)
    results = await graphiti.search_(query)
    print(f"✅Results: {results}")

    if not results:
        return {
            "query": query,
            "context": "",
            "sources": []
        }

    # -----------------------------
    # LLM-READY CONTEXT
    # -----------------------------
    context_blocks = []

    for idx, episode in enumerate(results.episodes, start=1):
        context_blocks.append(
            f"""
            [Source {idx}]
            Origin: {episode.source_description}

            Content:
            {episode.content}
            """.strip()
            )
    # print(f"❌ {context_blocks}")
    return {
        "query": query,
        "context": "\n\n".join(context_blocks)
    }

from langchain.agents import create_agent


# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

agent = create_agent(
    model=model,
    tools=[rag_tool,calculator],
    system_prompt="""
        You are a knowledgeable assistant.
        Use the provided context when answering.
        if the question is about document always call `rag_tool`
        `rag_tool` is an async so always invoke it as async
        """,
    debug=True,
    checkpointer=InMemorySaver()
)
# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
