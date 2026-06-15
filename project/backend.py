import sqlite3
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, List, Literal

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# =========================
# 1. STREAMING MODEL
# =========================
model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    streaming=True
)


# =========================
# 2. TOOLS
# =========================
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: Literal['add', 'sub', 'div', 'mul']) -> str:
    """Calculator tool to perform basic arithmetic operations.
    ## args:
    first_num: float
    second_num: float
    operation: literal['add', 'sub', 'mul', 'div']
    
    ## Return:
    result: str
    
    """
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        result = first_num / second_num if second_num != 0 else None
    else:
        return {"error": "invalid operation"}

    return result


tools = [search_tool, calculator]
tool_node = ToolNode(tools)

model_with_tools = model.bind_tools(tools)


# =========================
# 3. STATE
# =========================
class State(TypedDict):
    messages: Annotated[list, add_messages]


# =========================
# 4. CHAT NODE (CORE BRAIN)
# =========================
def chat_node(state: MessagesState):
    system_prompt = SystemMessage(
        content="You are a helpful gym trainer. Be concise and practical."
    )

    messages = [system_prompt] + state["messages"]

    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


# =========================
# 5. MEMORY (SQLite)
# =========================
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# =========================
# 6. BUILD GRAPH
# =========================
graph = StateGraph(MessagesState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    tools_condition
)

graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


