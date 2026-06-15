from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
load_dotenv()


GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

llm = init_chat_model("google_genai:gemini-2.5-pro")

# 🔹 Remote MCP client
client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()
    print(tools)
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        response = await  llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile()

async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    # content="Give me all my expenses from 1 Nov to 30 Nov"
                    content="hi"
                )
            ]
        }
    )

    print(result["messages"][-1])

if __name__ == "__main__":
    asyncio.run(main())
