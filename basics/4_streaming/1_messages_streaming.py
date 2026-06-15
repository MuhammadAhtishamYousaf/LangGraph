from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
def streaming(state: State) -> State:
    # print(state)
    messages = state.get('messages')
    llm_response = llm.invoke(messages)
    return {'messages': [llm_response]}


graph = StateGraph(State)

graph.add_node('streaming', streaming)

graph.add_edge(START, 'streaming')
graph.add_edge('streaming', END)

app = graph.compile()

question = HumanMessage(content = "what is langchain")

for chunk, metadata in app.stream({'messages': [question]}, stream_mode= 'messages'):
    if chunk.content:
        print(chunk.content, end = '', flush=True)