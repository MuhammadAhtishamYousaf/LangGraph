from typing import TypedDict, Annotated
import sqlite3
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', streaming = True)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
conn = sqlite3.connect(database= "sqlite_saver.db", timeout=10, check_same_thread=False)

sqlite_saver = SqliteSaver(conn=conn)
    
def chatbot(state: State) -> State:
    messages = state.get('messages')
    llm_response = llm.invoke(messages)
    return {'messages': [llm_response]}


graph = StateGraph(State)

graph.add_node('chatbot', chatbot)

graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)

app = graph.compile(checkpointer=sqlite_saver)

question = HumanMessage(content = "hi")

config = {'configurable': {'thread_id': '1'}}

for chunk, metadata in app.stream({'messages': [question]},config,  stream_mode='messages'):
    print(chunk.content, end = '', flush = True)
