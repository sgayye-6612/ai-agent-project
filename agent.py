import json
import os
import re
from typing import Annotated, TypedDict
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
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


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    return a / b


@tool
def get_current_time() -> str:
    """Get the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [
    calculator,
    multiply,
    subtract,
    divide,
    get_current_time,
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
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Scooby, a personal AI assistant.

Your name is Scooby.

Answer normally and naturally.

Do not use tools for greetings, names, personal information,
general conversation, or statements.

Tools should only be used when the user clearly asks for:
- addition
- multiplication
- subtraction
- division
- current time

Always answer the user after a tool has returned its result.
"""


# ============================================================
# 5. AGENT NODE
# ============================================================

def agent(state: State):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. NORMAL CHAT NODE
# ============================================================

def normal_chat(state: State):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 7. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools,
    handle_tool_errors=True
)


# ============================================================
# 8. DETECT TOOL REQUEST
# ============================================================

def needs_tool(text):

    text = text.lower().strip()

    # Time
    time_words = [
        "current time",
        "what time is it",
        "what's the time",
        "time now",
    ]

    for word in time_words:
        if word in text:
            return True

    # Must contain numbers for mathematical operations
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if len(numbers) >= 2:

        math_words = [
            "+",
            "add",
            "plus",
            "sum",
            "multiply",
            "multiplied",
            "times",
            "*",
            "subtract",
            "minus",
            "difference",
            "-",
            "divide",
            "divided",
            "quotient",
            "/",
        ]

        for word in math_words:
            if word in text:
                return True

    return False


# ============================================================
# 9. FIRST ROUTING
# ============================================================

def route_input(state: State):

    last_message = state["messages"][-1]

    if needs_tool(last_message.content):

        print("🔧 ROUTING → AGENT")

        return "agent"

    print("💬 ROUTING → NORMAL CHAT")

    return "normal"


# ============================================================
# 10. TOOL ROUTING
# ============================================================

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        print("🔧 ROUTING → TOOLS")

        return "tools"

    print("🏁 ROUTING → END")

    return "end"


# ============================================================
# 11. GRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node("agent", agent)
graph.add_node("normal", normal_chat)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    START,
    route_input,
    {
        "agent": "agent",
        "normal": "normal",
    },
)

graph.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "tools": "tools",
        "end": END,
    },
)

graph.add_edge(
    "tools",
    "agent"
)

graph.add_edge(
    "normal",
    END
)


# ============================================================
# 12. COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# 13. PERSISTENT MEMORY
# ============================================================

MEMORY_FILE = "memory.json"


def load_memory():

    if os.path.exists(MEMORY_FILE):

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    return []


def save_memory(memory):

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            memory,
            f,
            indent=2
        )


# ============================================================
# 14. LOAD MEMORY
# ============================================================

memory = load_memory()


# ============================================================
# 15. BUILD MESSAGE HISTORY
# ============================================================

messages = [
    SystemMessage(
        content=SYSTEM_PROMPT
    )
]


for item in memory:

    if item["role"] == "user":

        messages.append(
            HumanMessage(
                content=item["content"]
            )
        )

    elif item["role"] == "assistant":

        messages.append(
            AIMessage(
                content=item["content"]
            )
        )


# ============================================================
# 16. SCOOBY INTRO
# ============================================================

def is_scooby_intro(text):

    text = text.lower().strip()

    greetings = [
        "hi",
        "hello",
        "hey",
        "hey scooby",
        "hi scooby",
        "hello scooby",
    ]

    name_questions = [
        "what is your name",
        "what's your name",
        "whats your name",
        "who are you",
        "tell me your name",
    ]

    return (
        text in greetings
        or text in name_questions
    )


# ============================================================
# 17. CHAT LOOP
# ============================================================

while True:

    user_input = input("\nYou: ").strip()

    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

    if not user_input:

        print("Scooby: Please type something.")

        continue


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_input.lower() == "exit":

        save_memory(memory)

        print("Memory saved. 👋")

        break


    # --------------------------------------------------------
    # SCOOBY INTRO
    # --------------------------------------------------------

    if is_scooby_intro(user_input):

        response = (
            "Hi! My name is Scooby, your AI assistant. "
            "How can I help you?"
        )

        print("\nAI:", response)

        memory.append({
            "role": "user",
            "content": user_input
        })

        memory.append({
            "role": "assistant",
            "content": response
        })

        save_memory(memory)

        continue


    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    messages.append(
        HumanMessage(
            content=user_input
        )
    )


    # --------------------------------------------------------
    # RUN GRAPH
    # --------------------------------------------------------

    result = app.invoke(
        {
            "messages": messages
        }
    )


    # --------------------------------------------------------
    # UPDATE MESSAGES
    # --------------------------------------------------------

    messages = result["messages"]


    # --------------------------------------------------------
    # GET FINAL ANSWER
    # --------------------------------------------------------

    last_message = messages[-1]

    print("\nAI:", last_message.content)


    # --------------------------------------------------------
    # SAVE MEMORY
    # --------------------------------------------------------

    memory.append({
        "role": "user",
        "content": user_input
    })

    memory.append({
        "role": "assistant",
        "content": last_message.content
    })

    save_memory(memory)

