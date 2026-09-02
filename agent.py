import json
import os
import re
from typing import Annotated, TypedDict
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from rag_tool import search_notes


# ============================================================
# 1. STATE
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. TOOLS
# ============================================================

@tool
def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: float, b: float) -> float:
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
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Scooby, a personal AI assistant.

Your name is Scooby.

Be friendly, concise, and accurate.

For normal conversation, answer naturally.

For questions about the user's notes:
- Only use information provided by the retrieved notes.
- Never invent information.
- Keep answers short and direct.
"""


# ============================================================
# 5. NORMAL CHAT
# ============================================================

def normal_chat(state: State):

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


# ============================================================
# 6. RAG
# ============================================================

def rag_node(state: State):

    question = state["messages"][-1].content

    print("📚 SEARCHING NOTES...")

    try:
        context = search_notes.invoke(question)

    except Exception as e:

        return {
            "messages": [
                AIMessage(
                    content=f"Sorry, I couldn't search your notes. Error: {e}"
                )
            ]
        }

    if not context or context.strip() == "No relevant information found.":

        return {
            "messages": [
                AIMessage(
                    content="I couldn't find that information in your notes."
                )
            ]
        }

    prompt = f"""
Answer the user's question using ONLY the information in the notes.

Rules:
- Give only the direct answer.
- Do not say "Ruh-roh".
- Do not use the user's name unless the question asks for it.
- Do not add personality or jokes.
- Do not give advice.
- Do not explain unrelated information.
- Do not guess.
- If the answer is in the notes, answer it directly.
- Keep the answer to one short sentence whenever possible.

Notes:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            AIMessage(content=response.content.strip())
        ]
    }


# ============================================================
# 7. MATH DETECTION
# ============================================================

def parse_math(text: str):

    text = text.lower().strip()

    # --------------------------------------------------------
    # SYMBOL OPERATORS
    # --------------------------------------------------------

    pattern = r"(-?\d+(?:\.\d+)?)\s*(\+|-|\*|/)\s*(-?\d+(?:\.\d+)?)"

    match = re.search(pattern, text)

    if match:

        a = float(match.group(1))
        operator = match.group(2)
        b = float(match.group(3))

        return a, operator, b

    # --------------------------------------------------------
    # WORD OPERATORS
    # --------------------------------------------------------

    word_patterns = [
        (r"(-?\d+(?:\.\d+)?)\s+plus\s+(-?\d+(?:\.\d+)?)", "+"),
        (r"(-?\d+(?:\.\d+)?)\s+minus\s+(-?\d+(?:\.\d+)?)", "-"),
        (r"(-?\d+(?:\.\d+)?)\s+(?:times|multiplied by)\s+(-?\d+(?:\.\d+)?)", "*"),
        (r"(-?\d+(?:\.\d+)?)\s+(?:divided by|divide by)\s+(-?\d+(?:\.\d+)?)", "/"),
    ]

    for pattern, operator in word_patterns:

        match = re.search(pattern, text)

        if match:

            a = float(match.group(1))
            b = float(match.group(2))

            return a, operator, b

    return None


# ============================================================
# 8. MATH NODE
# ============================================================

def math_node(state: State):

    question = state["messages"][-1].content

    parsed = parse_math(question)

    if not parsed:

        return {
            "messages": [
                AIMessage(
                    content="I couldn't understand the calculation."
                )
            ]
        }

    a, operator, b = parsed

    print("🔢 MATH CALCULATION")

    # --------------------------------------------------------
    # ADDITION
    # --------------------------------------------------------

    if operator == "+":

        result = calculator.invoke({
            "a": a,
            "b": b
        })

    # --------------------------------------------------------
    # SUBTRACTION
    # --------------------------------------------------------

    elif operator == "-":

        result = subtract.invoke({
            "a": a,
            "b": b
        })

    # --------------------------------------------------------
    # MULTIPLICATION
    # --------------------------------------------------------

    elif operator == "*":

        result = multiply.invoke({
            "a": a,
            "b": b
        })

    # --------------------------------------------------------
    # DIVISION
    # --------------------------------------------------------

    elif operator == "/":

        if b == 0:

            return {
                "messages": [
                    AIMessage(
                        content="You can't divide by zero."
                    )
                ]
            }

        result = divide.invoke({
            "a": a,
            "b": b
        })

    else:

        return {
            "messages": [
                AIMessage(
                    content="Unsupported operation."
                )
            ]
        }

    # Make 30.0 display as 30
    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return {
        "messages": [
            AIMessage(
                content=f"The answer is {result}."
            )
        ]
    }


# ============================================================
# 9. TIME NODE
# ============================================================

def time_node(state: State):

    result = get_current_time.invoke({})

    return {
        "messages": [
            AIMessage(
                content=f"The current time is {result}."
            )
        ]
    }


# ============================================================
# 10. DETECT MATH
# ============================================================

def needs_math(text: str) -> bool:

    return parse_math(text) is not None


# ============================================================
# 11. DETECT TIME
# ============================================================

def needs_time(text: str) -> bool:

    text = text.lower().strip()

    time_phrases = [
        "current time",
        "what time is it",
        "what's the time",
        "whats the time",
        "time now",
        "what time",
    ]

    return any(phrase in text for phrase in time_phrases)


# ============================================================
# 12. DETECT RAG
# ============================================================

def needs_rag(text: str) -> bool:

    text = text.lower().strip()

    rag_phrases = [
        "my notes",
        "my note",
        "in my notes",
        "in my note",
        "according to my notes",
        "according to my note",
        "what does my note say",
        "what do my notes say",
        "what did i put in my notes",
        "what did i write in my notes",
        "my ai goal",
        "my goal",
        "what am i learning",
        "what programming language am i learning",
        "what language am i learning",
    ]

    return any(
        phrase in text
        for phrase in rag_phrases
    )


# ============================================================
# 13. SCOOBY INTRO
# ============================================================

def is_scooby_intro(text: str) -> bool:

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
# 14. ROUTING
# ============================================================

def route_input(state: State):

    text = state["messages"][-1].content

    # --------------------------------------------------------
    # MATH FIRST
    # --------------------------------------------------------

    if needs_math(text):

        print("🔢 ROUTING → MATH")

        return "math"

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if needs_time(text):

        print("🕐 ROUTING → TIME")

        return "time"

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if needs_rag(text):

        print("📚 ROUTING → RAG")

        return "rag"

    # --------------------------------------------------------
    # NORMAL CHAT
    # --------------------------------------------------------

    print("💬 ROUTING → NORMAL CHAT")

    return "normal"


# ============================================================
# 15. GRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node(
    "math",
    math_node
)

graph.add_node(
    "time",
    time_node
)

graph.add_node(
    "rag",
    rag_node
)

graph.add_node(
    "normal",
    normal_chat
)


# ============================================================
# START ROUTING
# ============================================================

graph.add_conditional_edges(
    START,
    route_input,
    {
        "math": "math",
        "time": "time",
        "rag": "rag",
        "normal": "normal",
    }
)


# ============================================================
# END ROUTES
# ============================================================

graph.add_edge(
    "math",
    END
)

graph.add_edge(
    "time",
    END
)

graph.add_edge(
    "rag",
    END
)

graph.add_edge(
    "normal",
    END
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# 16. MEMORY
# ============================================================

MEMORY_FILE = "memory.json"


def load_memory():

    if os.path.exists(MEMORY_FILE):

        try:

            with open(
                MEMORY_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        except Exception:

            return []

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
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# 17. LOAD MEMORY
# ============================================================

memory = load_memory()


# ============================================================
# 18. MESSAGE HISTORY
# ============================================================

messages = []

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
# 19. CHAT LOOP
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
    # UPDATE HISTORY
    # --------------------------------------------------------

    messages = result["messages"]

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    last_message = messages[-1]

    print(
        "\nAI:",
        last_message.content
    )

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