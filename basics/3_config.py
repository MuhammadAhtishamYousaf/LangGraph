from typing import TypedDict, Annotated
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {"configurable": {"thread_id": "thread-1"}}

# response = chatbot.invoke(
#     {'messages': [HumanMessage(content="Hi my name is ahtisham")]},
#     config=CONFIG
# )

print(chatbot.get_state(config=CONFIG).values['messages'])