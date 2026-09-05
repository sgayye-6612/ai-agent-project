from langchain_core.tools import tool
from langchain_ollama import ChatOllama

@tool
def reverse_text(text: str) -> str:
    """Reverse a piece of text."""
    return text[::-1]
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"


llm = ChatOllama(model="llama3.2")
llm_with_tools = llm.bind_tools([calculator, reverse_text])

response = llm_with_tools.invoke(
    "What is 25 multiplied by 4?"
)

print(response)
