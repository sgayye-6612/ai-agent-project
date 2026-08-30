from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from datetime import datetime

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
    try:
        return a + b
    except Exception as e:
        return f"Calculator error: {e}"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@tool
def get_current_time() -> str:
    """Get the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    
    return a / b

tools = [
    calculator,
    multiply,
    subtract,
    divide,
    get_current_time
]

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

tool_node = ToolNode(
    tools,
    handle_tool_errors=True
)


# ============================================================
# 6. ROUTING
# ============================================================
def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        print("🔧 ROUTING → TOOLS")
        return "tools"

    print("🏁 ROUTING → END")
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
# =========================
# CHAT LOOP
# =========================

messages = [
    SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use calculator only when the user asks for a mathematical calculation. "
            "Do not use tools for greetings or normal conversation."
        )
    )
]

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    result = app.invoke({
        "messages": messages
    })

    messages = result["messages"]

    last_message = messages[-1]

    print("AI:", last_message.content)


# ============================================================
# 10. RUN AGENT
# ============================================================

result = app.invoke(
    {
        "messages": [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Use calculator only when the user asks for a mathematical calculation. "
                    "Do not use tools for greetings or normal conversation."
                )
            ),
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