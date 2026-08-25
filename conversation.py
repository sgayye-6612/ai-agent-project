from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(model="llama3.2")

messages = []

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("AI:", response.content)