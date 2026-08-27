from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. TOOLS
# ============================================================

@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


tools = [calculator, multiply]


# ============================================================
# 3. LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 4. AGENT NODE
# ============================================================

def agent(state: State):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 5. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 6. ROUTING
# ============================================================

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "end"


# ============================================================
# 7. GRAPH
# ============================================================

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
    },
)

graph.add_edge("tools", "agent")


# ============================================================
# 8. COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# 9. USER INPUT
# ============================================================

user_input = input("You: ")


# ============================================================
# 10. RUN AGENT
# ============================================================

result = app.invoke(
    {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }
)


# ============================================================
# 11. PRINT CONVERSATION
# ============================================================

print("\n--- Conversation ---")

for i, message in enumerate(result["messages"], 1):

    print(f"\nMessage {i}")
    print("Type:", message.__class__.__name__)

    if message.content:
        print("Content:", message.content)

    if getattr(message, "tool_calls", None):
        print("Tool calls:", message.tool_calls)