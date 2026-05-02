import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; text: string };

const API_BASE = "http://localhost:8000";
const AGE_STORAGE_KEY = "shs_user_age";
const SESSION_STORAGE_KEY = "shs_session_id";

function getSessionId(): string {
  const existing = localStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) return existing;
  const next = Math.random().toString(36).slice(2, 12);
  localStorage.setItem(SESSION_STORAGE_KEY, next);
  return next;
}

function getInitialAge(): number {
  const stored = Number(localStorage.getItem(AGE_STORAGE_KEY));
  if (Number.isFinite(stored) && stored >= 14 && stored <= 100) {
    return stored;
  }
  return 18;
}

export function App() {
  const sessionId = useMemo(getSessionId, []);
  const [age, setAge] = useState<number>(getInitialAge);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [messages, setMessages] = useState<Msg[]>([
    // {
    //   role: "assistant",
    //   text: "Evidence-based secondhand smoke information to help you protect your child. Ask your question."
    // }
  ]);
  const chatRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const chat = chatRef.current;
    if (!chat) return;
    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    localStorage.setItem(AGE_STORAGE_KEY, String(age));
  }, [age]);

  async function onSend(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    setQuery("");
    setLoading(true);
    setStatus("Connecting...");
    setMessages((prev) => [...prev, { role: "user", text: q }, { role: "assistant", text: "" }]);

    const url = `${API_BASE}/chat/stream?session_id=${encodeURIComponent(sessionId)}&user_age=${age}&query=${encodeURIComponent(q)}`;
    const res = await fetch(url, { method: "GET", headers: { Accept: "text/event-stream" } });
    if (!res.ok || !res.body) {
      setLoading(false);
      setStatus("Error");
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", text: "Unable to connect to chat service." };
        return next;
      });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;

    while (!done) {
      const chunk = await reader.read();
      done = chunk.done;
      if (chunk.value) {
        buffer += decoder.decode(chunk.value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const eventLine = part.split("\n").find((l) => l.startsWith("event:"));
          const dataLine = part.split("\n").find((l) => l.startsWith("data:"));
          const event = eventLine?.replace("event:", "").trim();
          const dataRaw = dataLine?.replace("data:", "").trim() || "{}";
          const data = JSON.parse(dataRaw);

          if (event === "status") {
            setStatus(data.message || "Working...");
          } else if (event === "token") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { role: "assistant", text: last.text + (data.text || "") };
              }
              return next;
            });
          } else if (event === "error") {
            setStatus("Error");
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = { role: "assistant", text: data.message || "Request failed." };
              return next;
            });
          } else if (event === "done") {
            setStatus("Complete");
          }
        }
      }
    }

    setLoading(false);
  }

  function renderWithBold(text: string) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((p, i) => {
      if (p.startsWith("**") && p.endsWith("**") && p.length > 4) {
        return <strong key={i}>{p.slice(2, -2)}</strong>;
      }
      return <span key={i}>{p}</span>;
    });
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <img src="/shield-kid.svg" alt="SHS safety" className="brand-img" />
          <div className="brand-name">
            <h1>Smoke free parents-healthy kid</h1>
            <p>Evidence-based secondhand smoke information to help you protect your child.</p>
          </div>
        </div>
        {/* <div className="status">{status}</div> */}
      </header>

      {/* <section className="hero">
        <img src="/home-smoke-free.svg" alt="Smoke-free home" className="hero-img" />
        <div className="hero-copy">
          <h2>Protect your child with practical smoke-free actions</h2>
          <p>Ask one question at a time and get streamed guidance grounded in trusted health sources.</p>
        </div>
      </section> */}

      <main className="chat" ref={chatRef}>
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="msg-row">
              <div className={`avatar ${m.role}`}>{m.role === "user" ? "You" : "AI"}</div>
              <div className="msg-content">
                {m.text ? renderWithBold(m.text) : (loading && m.role === "assistant" ? <span className="typing">...</span> : "")}
              </div>
            </div>
          </div>
        ))}
      </main>

      {/* <section className="quick-prompts-section" aria-label="Quick prompts">
        <div className="section-heading">
          <span>Quick Prompts</span>
          <span className="section-hint">Swipe for more</span>
        </div>
        <div className="quick-prompts" role="list">
          {quickPrompts.map((p) => (
            <button
              type="button"
              key={p}
              className="chip"
              disabled={loading}
              onClick={() => setQuery(p)}
            >
              {p}
            </button>
          ))}
        </div>
      </section> */}

      <form className="composer" onSubmit={onSend}>
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about secondhand smoke, quitting, or child exposure..."
          maxLength={1500}
          disabled={loading}
        />
        <button aria-label="Send message" disabled={loading || !query.trim()} type="submit" className="send-icon-btn">
          {loading ? (
            <span className="spinner" />
          ) : (
            <svg viewBox="0 0 24 24" className="send-icon" aria-hidden="true">
              <path d="M3 20L22 12L3 4V10L15 12L3 14V20Z" />
            </svg>
          )}
        </button>
      </form>
    </div>
  );
}
