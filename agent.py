from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str
    result: int | None


# ----- TOOL -----

def calculator(a: int, b: int) -> int:
    return a + b


# ----- NODES -----

def agent_node(state: State):
    message = state["message"].lower()

    # Simulate the LLM deciding to use a tool
    if "add" in message or "calculate" in message:
        return {
            "message": message,
            "result": calculator(10, 20)
        }

    return {
        "message": message,
        "result": None
    }


def response_node(state: State):

    if state["result"] is not None:
        return {
            "message": f"The answer is {state['result']}",
            "result": state["result"]
        }

    return {
        "message": f"I received: {state['message']}",
        "result": None
    }


# ----- GRAPH -----

graph = StateGraph(State)

graph.add_node("agent", agent_node)
graph.add_node("response", response_node)

graph.add_edge(START, "agent")
graph.add_edge("agent", "response")
graph.add_edge("response", END)

app = graph.compile()


# ----- RUN -----

user_input = input("You: ")

result = app.invoke({
    "message": user_input,
    "result": None
})

print("Agent:", result["message"])