from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import ToolNode, tools_condition

from langchain_google_genai import ChatGoogleGenerativeAI

tavily_search=TavilySearch(max_results=3)
# tool.invoke("What is langgraph") 


## Custom function
def multiply(a:int, b:int) ->int:
    """Multiply a and b

    Args:
        a (int): first int
        b (int): second int

    Returns:
        int: output int
    """
    return a*b




llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

tools = [tavily_search, multiply]

llm_with_tools = llm.bind_tools(tools)
    
def chatbot(state: MessagesState) -> MessagesState:
    messages = state.get('messages')
    llm_response = llm_with_tools.invoke(messages)
    return {'messages': [llm_response]}


graph = StateGraph(MessagesState)

graph.add_node('chatbot', chatbot)
graph.add_node('tools', ToolNode(tools))

graph.add_edge(START, 'chatbot')
graph.add_conditional_edges('chatbot', tools_condition, {"tools": "tools", "__end__": "__end__"})
graph.add_edge('tools', 'chatbot')
graph.add_edge('chatbot', END)

app = graph.compile(checkpointer=MemorySaver())

question = HumanMessage("what is 5  times 5")

config = {'configurable': {'thread_id': '3'}}

for event in app.stream({'messages': [question]}, config, stream_mode='updates'):
    
    if 'chatbot' in event:
        msg = event['chatbot']['messages'][-1]
        
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"\n 🔧 Calling tools: {msg.tool_calls}")
        
        elif msg.content:
            print(f"\n Final Answer: ")
            print(msg.content)

    elif 'tools' in event:
        print("\n⚙️ Tool Result:")
        print(event["tools"]["messages"][-1].content)