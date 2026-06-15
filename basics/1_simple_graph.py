from typing import TypedDict

from langgraph.graph import StateGraph


class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State) -> State:
    state["answer"] = f"You asked: {state['question']}"
    return state

graph = StateGraph(State)

graph.add_node("answer", answer_node) #Nodes are actions

graph.set_entry_point("answer") #same as START
graph.set_finish_point("answer") # same as END

app = graph.compile()

result = app.invoke({"question": "What is LangGraph?"})

print(result)

# print(app.get_state().values['messages']) without config we can't access state