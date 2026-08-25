from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} like I am a beginner."
)

chain = prompt | llm

topic = input("What topic? ")

response = chain.invoke({"topic": topic})

print(response.content)