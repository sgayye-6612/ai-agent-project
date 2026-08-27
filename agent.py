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


def calculator_node(state: State):
    result = calculator(10, 20)

    return {
        "message": f"Calculator result: {result}"
    }


def normal_node(state: State):
    return {
        "message": f"Normal response: {state['message']}"
    }


# ----- DECISION -----

def decide_tool(state: State):

    message = state["message"].lower()

    if "calculate" in message or "add" in message:
        return "calculator"

    return "normal"


# ----- GRAPH -----

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("calculator", calculator_node)
graph.add_node("normal", normal_node)

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    decide_tool,
    {
        "calculator": "calculator",
        "normal": "normal"
    }
)

graph.add_edge("calculator", END)
graph.add_edge("normal", END)

app = graph.compile()


# ----- RUN -----

user_input = input("You: ")

result = app.invoke({
    "message": user_input
})

print("Agent:", result["message"])