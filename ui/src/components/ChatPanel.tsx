import { useState, useRef, useEffect } from 'react';
import { baseUrl } from '../api/http';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  source?: string;
}

interface ChatPanelProps {
  memoryId: string;
  patterns?: string[];
  context?: string;
}

export default function ChatPanel({ memoryId, patterns, context }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${baseUrl()}/agent/memory/${memoryId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-6).map(m => ({ role: m.role, content: m.content })),
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: data.reply || 'No response.',
        source: data.source,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (e: any) {
      setError(e.message || 'Failed to send message');
      // Remove the user message on error so they can retry
      setMessages(prev => prev.slice(0, -1));
      setInput(text);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, overflow: 'hidden', marginTop: 12 }}>
      {/* Header */}
      <div style={{
        padding: '8px 12px',
        background: 'rgba(255,255,255,0.04)',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        fontSize: 12,
        fontWeight: 600,
      }}>
        <span>💬</span>
        <span>Ask about this event</span>
        {patterns && patterns.length > 0 && (
          <span style={{ marginLeft: 'auto', fontSize: 10, opacity: 0.5 }}>
            {patterns.length} pattern{patterns.length > 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Messages area */}
      <div
        ref={scrollRef}
        style={{
          height: messages.length > 0 ? 200 : 60,
          overflowY: 'auto',
          padding: 8,
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          transition: 'height 0.2s',
        }}
      >
        {messages.length === 0 && (
          <div style={{ opacity: 0.4, fontSize: 11, textAlign: 'center', padding: 12 }}>
            Ask "Why was this flagged?" or "What should I do?"
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
              padding: '6px 10px',
              borderRadius: 8,
              fontSize: 11,
              lineHeight: 1.4,
              background: msg.role === 'user'
                ? 'rgba(52,152,219,0.2)'
                : 'rgba(255,255,255,0.06)',
              border: msg.role === 'user'
                ? '1px solid rgba(52,152,219,0.3)'
                : '1px solid rgba(255,255,255,0.08)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {msg.content}
            {msg.source && msg.role === 'assistant' && (
              <span style={{
                display: 'inline-block',
                marginLeft: 6,
                fontSize: 9,
                padding: '1px 4px',
                borderRadius: 3,
                background: msg.source === 'llm' ? 'rgba(46,204,113,0.2)' :
                            msg.source === 'rag' ? 'rgba(155,89,182,0.2)' :
                            'rgba(255,255,255,0.1)',
                color: msg.source === 'llm' ? '#2ecc71' :
                       msg.source === 'rag' ? '#9b59b6' : '#999',
              }}>
                {msg.source}
              </span>
            )}
          </div>
        ))}
        {loading && (
          <div style={{
            alignSelf: 'flex-start',
            padding: '6px 10px',
            fontSize: 11,
            opacity: 0.5,
            fontStyle: 'italic',
          }}>
            Analyzing…
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div style={{
          padding: '4px 12px',
          fontSize: 10,
          color: 'var(--danger, #e74c3c)',
          background: 'rgba(231,76,60,0.08)',
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Input area */}
      <div style={{
        display: 'flex',
        borderTop: '1px solid rgba(255,255,255,0.08)',
      }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask the AI analyst…"
          disabled={loading}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: 'inherit',
            fontSize: 11,
            padding: '8px 12px',
          }}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          style={{
            background: 'transparent',
            border: 'none',
            color: loading || !input.trim() ? 'rgba(255,255,255,0.2)' : 'var(--accent, #3498db)',
            cursor: loading || !input.trim() ? 'default' : 'pointer',
            padding: '8px 12px',
            fontSize: 13,
          }}
        >
          ➤
        </button>
      </div>
    </div>
  );
}
