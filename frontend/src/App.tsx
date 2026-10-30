
import { useRef, useState, type KeyboardEvent, type FormEvent } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Sparkles } from "@react-three/drei";
import type { Group } from "three";
import { motion, AnimatePresence } from "motion/react";
import {
  Bot, Send, Plus, Sparkles as SparkleIcon,
  UserRound, Trash2, Zap, MessageCircle
} from "lucide-react";
import "./App.css";

type Message = {
  role: "user" | "assistant";
  content: string;
};

function Orb() {
  const orb = useRef<Group>(null);

  useFrame(({ clock, pointer }) => {
    if (!orb.current) return;
    orb.current.rotation.y = clock.elapsedTime * 0.25 + pointer.x * 0.15;
    orb.current.rotation.x = Math.sin(clock.elapsedTime * 0.35) * 0.15;
    orb.current.position.y = Math.sin(clock.elapsedTime * 0.8) * 0.12;
  });

  return (
    <Float speed={2} rotationIntensity={0.3} floatIntensity={0.5}>
      <group ref={orb}>
        <mesh>
          <icosahedronGeometry args={[1, 5]} />
          <meshStandardMaterial
            color="#9474ff"
            emissive="#6335df"
            emissiveIntensity={1.5}
            metalness={0.7}
            roughness={0.18}
          />
        </mesh>
        <mesh scale={1.18}>
          <icosahedronGeometry args={[1, 2]} />
          <meshBasicMaterial
            color="#b9a5ff"
            wireframe
            transparent
            opacity={0.35}
          />
        </mesh>
        <mesh scale={1.38}>
          <icosahedronGeometry args={[1, 1]} />
          <meshBasicMaterial
            color="#7545ef"
            wireframe
            transparent
            opacity={0.15}
          />
        </mesh>
      </group>
    </Float>
  );
}

function AssistantOrb() {
  return (
    <div className="orb-scene">
      <Canvas camera={{ position: [0, 0, 4], fov: 45 }}>
        <ambientLight intensity={0.8} />
        <pointLight position={[3, 3, 3]} intensity={30} color="#a98aff" />
        <pointLight position={[-3, -2, 2]} intensity={15} color="#5132ff" />
        <Orb />
        <Sparkles count={100} scale={4} size={2} speed={0.4} color="#bba8ff" />
      </Canvas>
    </div>
  );
}

const suggestions = [
  "What can you help me with?",
  "Calculate 10 + 20",
  "What am I learning?",
  "Explain artificial intelligence"
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || loading) return;

    setMessages(prev => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question })
      });

      if (!response.ok) throw new Error("Backend request failed");

      const data = await response.json();
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: data.response }
      ]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: "I couldn't connect to Scooby's backend. Please check that your AI server is running."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    sendMessage();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const newChat = () => {
    setMessages([]);
    setInput("");
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Bot size={23} /></div>
          <div>
            <h2>SCOOBY<span>.</span></h2>
            <small>PERSONAL AI</small>
          </div>
        </div>

        <button className="new-chat" onClick={newChat}>
          <Plus size={18} /> New conversation
        </button>

        <div className="side-label">WORKSPACE</div>
        <div className="side-link active"><MessageCircle size={17} /> AI Assistant</div>
        <div className="side-link"><SparkleIcon size={17} /> Explore AI</div>

        <div className="sidebar-bottom">
          <div className="profile">
            <div className="profile-avatar">S</div>
            <div><strong>My Workspace</strong><small>Personal assistant</small></div>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace /</span>
            <strong> AI Assistant</strong>
          </div>
          <div className="online-status">
            <span className="online-dot" /> Systems ready
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <motion.div
              className="welcome"
              initial={{ opacity: 0, y: 25 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
            >
              <div className="orb-wrap">
                <AssistantOrb />
                <div className="orb-glow" />
              </div>

              <div className="welcome-tag">
                <SparkleIcon size={13} /> YOUR INTELLIGENT COMPANION
              </div>

              <h1>Meet <span>Scooby.</span></h1>
              <p className="welcome-description">
                A little curiosity can take you anywhere.
                Ask me something and let's explore together.
              </p>

              <div className="suggestion-grid">
                {suggestions.map((suggestion, index) => (
                  <motion.button
                    key={suggestion}
                    className="suggestion-card"
                    onClick={() => setInput(suggestion)}
                    whileHover={{ y: -4, scale: 1.02 }}
                    whileTap={{ scale: 0.97 }}
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + index * 0.1 }}
                  >
                    <SparkleIcon size={16} />
                    <span>{suggestion}</span>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          ) : (
            <div className="messages">
              <AnimatePresence initial={false}>
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    className={`message-row ${message.role}`}
                    initial={{ opacity: 0, y: 14, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className={`message-avatar ${message.role}`}>
                      {message.role === "user"
                        ? <UserRound size={18} />
                        : <Bot size={19} />}
                    </div>
                    <div className="message-content">
                      <span className="message-name">
                        {message.role === "user" ? "You" : "Scooby"}
                      </span>
                      <div className="message-bubble">{message.content}</div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {loading && (
                <motion.div className="message-row assistant" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <div className="message-avatar assistant"><Bot size={19} /></div>
                  <div className="message-content">
                    <span className="message-name">Scooby</span>
                    <div className="message-bubble thinking">
                      <span /><span /><span /> Thinking...
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          )}
        </section>

        <footer className="composer-area">
          <form className="composer" onSubmit={handleSubmit}>
            <SparkleIcon className="composer-sparkle" size={19} />
            <input
              value={input}
              onChange={event => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask Scooby anything..."
              disabled={loading}
            />
            <motion.button
              type="submit"
              className="send-button"
              disabled={!input.trim() || loading}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              aria-label="Send message"
            >
              <Send size={18} />
            </motion.button>
          </form>
          <div className="composer-footer">
            <span><Zap size={12} /> Powered by your personal AI agent</span>
            <span><Trash2 size={12} /> <button onClick={newChat}>Clear conversation</button></span>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
