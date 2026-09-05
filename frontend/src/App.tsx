import { useState } from "react";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
        }),
      });

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I couldn't connect to Scooby's backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>🐶 Scooby</h1>
          <p>Your AI Assistant</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Online
        </div>
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="dog">🐶</div>

            <h2>Hi! I'm Scooby.</h2>

            <p>
              Your personal AI assistant. Ask me anything!
            </p>

            <div className="suggestions">
              <button
                onClick={() => setInput("What is your name?")}
              >
                What's your name?
              </button>

              <button
                onClick={() => setInput("10 + 20")}
              >
                Calculate 10 + 20
              </button>

              <button
                onClick={() => setInput("What am I learning?")}
              >
                What am I learning?
              </button>
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message-row ${message.role}`}
              >
                <div className="avatar">
                  {message.role === "user" ? "👤" : "🐶"}
                </div>

                <div className="message">
                  {message.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-row assistant">
                <div className="avatar">🐶</div>

                <div className="message typing">
                  Scooby is thinking...
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <div className="input-area">
        <div className="input-box">
          <input
            type="text"
            placeholder="Message Scooby..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />

          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
          >
            Send
          </button>
        </div>

        <p className="footer-text">
          Scooby can make mistakes. Check important information.
        </p>
      </div>
    </div>
  );
}

export default App;