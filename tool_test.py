from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [multiply]

tool_node = ToolNode(tools)


ai_message = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "multiply",
            "args": {
                "a": 10,
                "b": 20
            },
            "id": "call_1",
            "type": "tool_call",
        }
    ],
)


result = tool_node.invoke({
    "messages": [ai_message]
})


print(result)