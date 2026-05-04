import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; text: string };
type ChatSession = { id: string; title: string; messages: Msg[]; updatedAt: number };

const API_BASE = "http://localhost:8000";
const AGE_STORAGE_KEY = "shs_user_age";
const CHAT_SESSIONS_KEY = "shs_chat_sessions";
const ACTIVE_SESSION_KEY = "shs_active_session";
const THEME_KEY = "shs_theme";

function newSessionId(): string {
  return Math.random().toString(36).slice(2, 12);
}

function getInitialAge(): number {
  const stored = Number(localStorage.getItem(AGE_STORAGE_KEY));
  if (Number.isFinite(stored) && stored >= 14 && stored <= 100) return stored;
  return 18;
}

function getInitialTheme(): "dark" | "light" {
  const t = localStorage.getItem(THEME_KEY);
  return t === "light" ? "light" : "dark";
}

function loadSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(CHAT_SESSIONS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatSession[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function App() {
  const [age, setAge] = useState<number>(getInitialAge);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [theme, setTheme] = useState<"dark" | "light">(getInitialTheme);
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions);
  const [activeSessionId, setActiveSessionId] = useState<string>(() => localStorage.getItem(ACTIVE_SESSION_KEY) || "");

  const chatRef = useRef<HTMLElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const activeSession = useMemo(() => sessions.find((s) => s.id === activeSessionId), [sessions, activeSessionId]);
  const messages = activeSession?.messages ?? [];

  useEffect(() => {
    if (sessions.length === 0) {
      const id = newSessionId();
      const first: ChatSession = { id, title: "New chat", messages: [], updatedAt: Date.now() };
      setSessions([first]);
      setActiveSessionId(id);
      return;
    }
    if (!activeSessionId || !sessions.some((s) => s.id === activeSessionId)) {
      setActiveSessionId(sessions[0].id);
    }
  }, [sessions, activeSessionId]);

  useEffect(() => {
    localStorage.setItem(AGE_STORAGE_KEY, String(age));
  }, [age]);

  useEffect(() => {
    localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    if (activeSessionId) localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);
  }, [activeSessionId]);

  useEffect(() => {
    localStorage.setItem(THEME_KEY, theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const chat = chatRef.current;
    if (!chat) return;
    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [query]);

  function createNewChat() {
    const id = newSessionId();
    const next: ChatSession = { id, title: "New chat", messages: [], updatedAt: Date.now() };
    setSessions((prev) => [next, ...prev]);
    setActiveSessionId(id);
    setStatus("Ready");
    setQuery("");
  }

  function updateActiveMessages(nextMessages: Msg[]) {
    setSessions((prev) =>
      prev.map((s) => {
        if (s.id !== activeSessionId) return s;
        const firstUser = nextMessages.find((m) => m.role === "user" && m.text.trim());
        const title = firstUser ? firstUser.text.slice(0, 44) : s.title;
        return { ...s, messages: nextMessages, title, updatedAt: Date.now() };
      })
    );
  }

  async function onSend(e: FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading || !activeSessionId) return;

    setQuery("");
    setLoading(true);
    setStatus("Connecting...");

    const seed = [...messages, { role: "user", text: q } as Msg, { role: "assistant", text: "" } as Msg];
    updateActiveMessages(seed);

    const url = `${API_BASE}/chat/stream?session_id=${encodeURIComponent(activeSessionId)}&user_age=${age}&query=${encodeURIComponent(q)}`;

    try {
      const res = await fetch(url, { method: "GET", headers: { Accept: "text/event-stream" } });
      if (!res.ok || !res.body) {
        setStatus("Error");
        updateActiveMessages([...seed.slice(0, -1), { role: "assistant", text: "Unable to connect to chat service." }]);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done = false;
      let assistantText = "";

      while (!done) {
        const chunk = await reader.read();
        done = chunk.done;
        if (!chunk.value) continue;

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
            assistantText += data.text || "";
            updateActiveMessages([...seed.slice(0, -1), { role: "assistant", text: assistantText }]);
          } else if (event === "error") {
            setStatus("Error");
            updateActiveMessages([...seed.slice(0, -1), { role: "assistant", text: data.message || "Request failed." }]);
          } else if (event === "done") {
            setStatus("Complete");
          }
        }
      }
    } catch {
      setStatus("Error");
      updateActiveMessages([...seed.slice(0, -1), { role: "assistant", text: "Network error while contacting the API." }]);
    } finally {
      setLoading(false);
    }
  }

  function onInputKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void onSend(e as unknown as FormEvent);
    }
  }

  function renderWithBold(text: string) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((p, i) => (p.startsWith("**") && p.endsWith("**") && p.length > 4 ? <strong key={i}>{p.slice(2, -2)}</strong> : <span key={i}>{p}</span>));
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-top">
          <button type="button" className="new-chat-btn" onClick={createNewChat}>+ New chat</button>
          <button type="button" className="theme-btn" onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}>
            Theme: {theme === "dark" ? "Dark" : "Light"}
          </button>
          <div className="history">
            {sessions
              .slice()
              .sort((a, b) => b.updatedAt - a.updatedAt)
              .map((s) => (
                <button key={s.id} type="button" className={`history-item ${s.id === activeSessionId ? "active" : ""}`} onClick={() => setActiveSessionId(s.id)}>
                  {s.title || "New chat"}
                </button>
              ))}
          </div>
        </div>
        <div className="sidebar-bottom">
          <span>Status: {status}</span>
          <span>Age: {age}+</span>
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <h1>SHS Cessation Assistant</h1>
        </header>

        <main className="chat" ref={chatRef}>
          {messages.length === 0 && (
            <div className="empty-state">
              <p>Try: "How can I reduce smoke exposure for my baby at home?"</p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`row ${m.role}`}>
              <div className="avatar">{m.role === "assistant" ? "AI" : "You"}</div>
              <div className="msg-content">{m.text ? renderWithBold(m.text) : loading && m.role === "assistant" ? <span className="typing">...</span> : ""}</div>
            </div>
          ))}
        </main>

        <form className="composer" onSubmit={onSend}>
          <textarea
            ref={inputRef}
            className="input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onInputKeyDown}
            placeholder="Message SHS Cessation Assistant..."
            maxLength={1500}
            disabled={loading}
            rows={1}
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
          <div className="composer-meta">
            <span>Status: {status}</span>
            <span>Age filter: {age}+</span>
          </div>
        </form>
      </section>
    </div>
  );
}
