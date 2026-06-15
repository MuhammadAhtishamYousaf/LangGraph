import asyncio
import os, json
from datetime import datetime, timezone

from graphiti_core.utils.maintenance.graph_data_operations import clear_data
# Load environment variables from .env file
from dotenv import load_dotenv

# Core Graphiti class
from graphiti_core import Graphiti

# Enum to tell Graphiti what type of data we are adding
from graphiti_core.nodes import EpisodeType

# Load environment variables
load_dotenv()


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# -----------------------------
# ENVIRONMENT CONFIGURATION
# -----------------------------

# Neo4j Aura connection URI (cloud)
NEO4J_URI = os.environ["NEO4J_URI"]

# Neo4j username (usually 'neo4j' for Aura)
NEO4J_USER = os.environ["NEO4J_USER"]

# Neo4j database password
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# -----------------------------
# MAIN ASYNC FUNCTION
# -----------------------------
graphiti = Graphiti(
    NEO4J_URI, 
    NEO4J_USER, 
    NEO4J_PASSWORD
    )

file = 'ML Roadmap.pdf'


# PDF_FILE = "ML Roadmap.pdf"

# -----------------------------
# PROCESS PDF → GRAPHITI
# -----------------------------
async def process_doc(file_path: str):
    # 1️⃣ Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2️⃣ Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3️⃣ Convert chunks → Graphiti episodes
    for i, doc in enumerate(chunks):
        await graphiti.add_episode(
            name=f"pdf_chunk_{i}",
            episode_body=doc.page_content,
            source=EpisodeType.text,
            source_description=f"PDF: {file_path}",
            reference_time=datetime.now(timezone.utc),
        )

        print(f"✅ Stored chunk {i}")

    print("🎯 All PDF chunks stored in Graphiti")

    await graphiti.close()
# asyncio.run(process_doc(file))

async def clear_indices():
    """
    Main async entry point.
    Everything in Graphiti is async because Neo4j driver is async.
    """
    await clear_data(graphiti.driver)              # optional for a fresh start
    await graphiti.build_indices_and_constraints() # creates indexes once
#     print(f" ✅Graphiti initiialized")
#     try:
        
        
        

#         # --------------------------------
#         # Run a SIMPLE search
#         # --------------------------------
#         # This performs:
#         # - semantic search
#         # - keyword search
#         # - graph-aware ranking

from typing import Optional
from langchain_core.tools import tool


async def rag_tool(query: str) -> dict:
    """
    Retrieve grounded, source-aware context from Graphiti
    for accurate LLM responses.
    """

    # Graphiti search (sync wrapper assumed)
    results = await graphiti.search_(query)

    if not results or not results.episodes:
        return {
            "query": query,
            "context": "",
            "sources": []
        }

    # -----------------------------
    # LLM-READY CONTEXT
    # -----------------------------
    context_blocks = []
    sources = []

    for idx, episode in enumerate(results.episodes, start=1):
        context_blocks.append(
            f"""
            [Source {idx}]
            Origin: {episode.source_description}

            Content:
            {episode.content}
            """.strip()
            )

    return {
        "query": query,
        "context": "\n\n".join(context_blocks)
    }

    # Print each retrieved fact
    # for result in results:
    #     print("Fact:", result.fact)
    #     print("Source node UUID:", result.source_node_uuid)
    #     print("---")

# print(asyncio.run(rag_tool("what is this document about")))
asyncio.run(clear_indices())                    