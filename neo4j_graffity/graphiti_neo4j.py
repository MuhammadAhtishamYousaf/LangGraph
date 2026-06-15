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

async def main():
    """
    Main async entry point.
    Everything in Graphiti is async because Neo4j driver is async.
    """

    # --------------------------------
    # Initialize Graphiti
    # --------------------------------
    # This:
    # - connects to Neo4j
    # - creates indexes if needed
    # - prepares graph schema


    graphiti = Graphiti(
        NEO4J_URI, 
        NEO4J_USER, 
        NEO4J_PASSWORD
        )
    # await clear_data(graphiti.driver)              # optional for a fresh start
    # await graphiti.build_indices_and_constraints() # creates indexes once
    print(f" ✅Graphiti initiialized")
    try:
        
        # --------------------------------
        # Add a SINGLE test episode
        # --------------------------------
        # Episode = raw knowledge input
        # Graphiti will:
        # - extract entities
        # - extract relationships
        # - store them as nodes + edges
        # episodes = [
        #     {
        #         "name": "About Me",
        #         "content": "Hi, I'm Pradip Nichite. I am the founder and CEO of FutureSmart AI.",
        #         "type": EpisodeType.text,
        #         "description": "intro"
        #     },
        #     {
        #         "name": "About FutureSmart AI",
        #         "content": "FutureSmart AI builds custom AI solutions for clients.",
        #         "type": EpisodeType.text,
        #         "description": "company overview"
        #     }
        # ]
        
        # for episode in episodes:
        #     await graphiti.add_episode(
        #     name=episode["name"],
        #     episode_body=episode["content"],
        #     source=episode["type"],
        #     source_description=episode["description"],
        #     reference_time=datetime.now(timezone.utc),
        # )
        #     print(f"✅ Episode added successfully {episode}")
        #     # helper loops and calls graphiti.add_episode()


        # --------------------------------
        # Run a SIMPLE search
        # --------------------------------
        # This performs:
        # - semantic search
        # - keyword search
        # - graph-aware ranking
        results = await graphiti.search(
            "what is user_fact"
        )

        print("\n🔍 Search Results:\n")

        # Print each retrieved fact
        for result in results:
            print("Fact:", result.fact)
            print("Source node UUID:", result.source_node_uuid)
            print("---")

    finally:
        # --------------------------------
        # Always close Graphiti connection
        # --------------------------------
        # This safely closes Neo4j async sessions
        await graphiti.close()
        print("\n🔒 Connection closed")

# -----------------------------
# SCRIPT ENTRY POINT
# -----------------------------

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
