\# Scooby — AI Personal Assistant



Scooby is a personal AI assistant built with Python, FastAPI, LangChain, LangGraph, and React. It provides a conversational interface for interacting with an AI agent and exploring assistant capabilities such as question answering, calculations, and retrieval-augmented generation (RAG).



\## Screenshots



\### Welcome Screen

!\[Scooby Welcome Screen](docs/screenshots/welcome-screen.png)



\### Chat Interface

!\[Scooby Chat Interface](docs/screenshots/chat-interface.png)



\### Assistant Response

!\[Scooby Assistant Response](docs/screenshots/assistant-response.png)



\### Additional View

!\[Scooby Additional View](docs/screenshots/additional-view.png)



\## Features



\- Conversational AI assistant

\- React and TypeScript frontend

\- FastAPI backend

\- LangChain and LangGraph agent integration

\- Retrieval-augmented generation (RAG)

\- Local LLM support using Ollama

\- Chroma vector database for knowledge retrieval

\- API documentation through Swagger UI



\## Tech Stack



\*\*Frontend\*\*

\- React

\- TypeScript

\- Vite

\- CSS

\- Three.js / React Three Fiber



\*\*Backend\*\*

\- Python

\- FastAPI

\- LangChain

\- LangGraph

\- Ollama

\- ChromaDB



\## Project Structure



```text

ai-agent-project/

├── backend/

│   └── main.py

├── frontend/

│   ├── src/

│   ├── package.json

│   └── index.html

├── docs/

│   └── screenshots/

├── agent.py

├── rag.py

├── rag\_tool.py

├── notes.txt

└── README.md

```



\## Getting Started



\### 1. Clone the repository



```bash

git clone YOUR\_REPOSITORY\_URL

cd ai-agent-project

```



\### 2. Set up the Python environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install backend dependencies:



```bash

pip install -r requirements.txt

```



\### 3. Start the backend



```bash

cd backend

uvicorn main:app --reload

```



Backend: `http://127.0.0.1:8000`



API documentation: `http://127.0.0.1:8000/docs`



\### 4. Start the frontend



Open another terminal:



```bash

cd frontend

npm install

npm run dev

```



Open the local URL displayed by Vite.



\## Future Improvements



\- Persistent conversation history

\- Improved RAG document ingestion

\- Additional AI tools and agent capabilities

\- Enhanced 3D assistant interactions

\- Better error handling and deployment support



\## Author



\*\*Sahithi Gayye\*\*



AI Engineer | Full Stack Developer



\[GitHub](https://github.com/sgayye-6612)

