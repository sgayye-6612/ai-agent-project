from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    message: str


# ----- TOOL -----

def calculator(a: int, b: int):
    return a + b


# ----- NODES -----

def chatbot(state: State):
    return {
        "message": state["message"]
    }


def tool_node(state: State):
    result = calculator(10, 20)

    return {
        "message": f"Calculator result: {result}"
    }


# ----- GRAPH -----

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("calculator", tool_node)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "calculator")
graph.add_edge("calculator", END)

app = graph.compile()


# ----- RUN -----

user_input = input("You: ")

result = app.invoke({
    "message": user_input
})

print("Agent:", result["message"])