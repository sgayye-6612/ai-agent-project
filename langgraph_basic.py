from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


class State(TypedDict):
    message: str
    route: str


llm = ChatOllama(model="llama3.2")


def classify(state: State):
    response = llm.invoke(
        f"""
        Classify the user's message as either:
        greeting
        or
        question

        User message: {state["message"]}

        Reply with only one word: greeting or question.
        """
    )

    return {"route": response.content.strip().lower()}


def greeting(state: State):
    return {"message": "Hello! Nice to meet you!"}


def answer(state: State):
    response = llm.invoke(state["message"])
    return {"message": response.content}


def route(state: State):
    if "greeting" in state["route"]:
        return "greeting"

    return "answer"


graph = StateGraph(State)

graph.add_node("classify", classify)
graph.add_node("greeting", greeting)
graph.add_node("answer", answer)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route,
    {
        "greeting": "greeting",
        "answer": "answer"
    }
)

graph.add_edge("greeting", END)
graph.add_edge("answer", END)

app = graph.compile()

question = input("You: ")

result = app.invoke({
    "message": question,
    "route": ""
})

print("\nAI:", result["message"])