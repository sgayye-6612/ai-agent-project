from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    message: str


def chatbot(state: State):
    return {
        "message": state["message"]
    }


def response_node(state: State):
    return {
        "message": f"Normal response: {state['message']}"
    }


def question_node(state: State):
    return {
        "message": f"Question detected: {state['message']}"
    }


# Decision function
def decide_next(state: State):
    message = state["message"]

    if "?" in message:
        return "question"

    return "normal"


graph = StateGraph(State)

# Nodes
graph.add_node("chatbot", chatbot)
graph.add_node("response", response_node)
graph.add_node("question", question_node)

# Start
graph.add_edge(START, "chatbot")

# Conditional edge
graph.add_conditional_edges(
    "chatbot",
    decide_next,
    {
        "question": "question",
        "normal": "response"
    }
)

# End
graph.add_edge("question", END)
graph.add_edge("response", END)

app = graph.compile()


user_input = input("You: ")

result = app.invoke({
    "message": user_input
})

print("Agent:", result["message"])