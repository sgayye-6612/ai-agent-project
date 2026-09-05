from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os

from langchain_core.messages import HumanMessage, AIMessage

# ------------------------------------------------------------
# Allow Python to find agent.py in the project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ------------------------------------------------------------
# Import Scooby
# ------------------------------------------------------------

from agent import app as scooby_app


# ------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------

app = FastAPI(
    title="Scooby AI Assistant API",
    description="Backend API for Scooby AI Assistant",
    version="1.0.0",
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Request Model
# ------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


# ------------------------------------------------------------
# Conversation History
# ------------------------------------------------------------

chat_history = []


# ------------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Scooby AI Assistant API is running!"
    }


# ------------------------------------------------------------
# Chat Endpoint
# ------------------------------------------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    user_message = request.message.strip()

    if not user_message:

        return {
            "response": "Please type a message."
        }

    # Add user's message to conversation history
    chat_history.append(
        HumanMessage(
            content=user_message
        )
    )

    # Send complete conversation to Scooby
    result = scooby_app.invoke(
        {
            "messages": chat_history
        }
    )

    # Get Scooby's response
    last_message = result["messages"][-1]

    # Update conversation history
    chat_history.clear()
    chat_history.extend(
        result["messages"]
    )

    return {
        "response": last_message.content
    }