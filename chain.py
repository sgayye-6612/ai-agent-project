from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    "You are a helpful teacher. Explain {topic} in simple terms."
)

chain = prompt | llm

while True:
    topic = input("Topic: ")

    if topic.lower() == "exit":
        break

    response = chain.invoke({"topic": topic})

    print("\nAI:", response.content)