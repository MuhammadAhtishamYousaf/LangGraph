from typing import Annotated
from typing_extensions import TypedDict


from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


def chatbot(state: ChatState) -> ChatState:
    """Node receives state, do one job, and return updated state"""

    response = llm.invoke(state["messages"])

    return {"messages": [response]}


builder = StateGraph(state_schema = ChatState)

builder.add_node(node='chatbot', action=chatbot)

builder.add_edge(start_key=START, end_key='chatbot')
builder.add_edge(start_key='chatbot', end_key=END)

graph: CompiledStateGraph  = builder.compile()

result: ChatState = graph.invoke(
    input={
        "messages": [
            HumanMessage(content="Hello")
        ]
    }
)

print(result['messages'][-1].content)

