from typing import TypedDict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


class State(TypedDict):
    messages: list


# ---------- TOOL ----------

@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


tools = [calculator]


# ---------- AGENT NODE ----------

def agent(state: State):
    message = state["messages"][-1]

    # Simulate the AI deciding to use the calculator
    if "add" in message.content.lower():

        ai_message = AIMessage(
            content="I need to use the calculator.",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {
                        "a": 10,
                        "b": 20
                    },
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )

        return {
            "messages": [
                message,
                ai_message
            ]
        }

    # No tool needed
    return {
        "messages": [
            message,
            AIMessage(
                content="I can answer without using a tool."
            )
        ]
    }


# ---------- TOOL NODE ----------

tool_node = ToolNode(tools)


# ---------- ROUTING ----------

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"


# ---------- GRAPH ----------

graph = StateGraph(State)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "tools": "tools",
        "end": END,
    }
)

graph.add_edge("tools", END)


# ---------- COMPILE ----------

app = graph.compile()


# ---------- RUN ----------

user_input = input("You: ")

result = app.invoke({
    "messages": [
        HumanMessage(content=user_input)
    ]
})


# ---------- PRINT MESSAGES ----------

print("\n--- Conversation ---")

for message in result["messages"]:
    print(message)