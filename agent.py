from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    message: str


@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def chatbot(state: State):
    return {
        "message": state["message"]
    }


def calculator_node(state: State):
    result = calculator.invoke({
        "a": 10,
        "b": 20
    })

    return {
        "message": f"Calculator result: {result}"
    }


graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("calculator", calculator_node)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", "calculator")
graph.add_edge("calculator", END)

app = graph.compile()


user_input = input("You: ")

result = app.invoke({
    "message": user_input
})

print("Agent:", result["message"])