import { useEffect, useRef, useState } from 'react';
import { Mic, MicOff, Send, FileText } from 'lucide-react';

const WS_URL = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000') + '/ws/chat?tenant=default';

interface Citation { index: number; document_id: number; chunk_id: number; text: string; score: number }
interface Message { role: 'user' | 'assistant'; text: string; citations?: Citation[] }

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState('');
  const [recording, setRecording] = useState(false);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sock = new WebSocket(WS_URL);
    sock.onopen = () => setConnected(true);
    sock.onclose = () => setConnected(false);
    sock.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'token') {
        setMessages((m) => {
          const last = m[m.length - 1];
          if (last?.role === 'assistant') {
            return [...m.slice(0, -1), { ...last, text: last.text + data.text }];
          }
          return [...m, { role: 'assistant', text: data.text }];
        });
      } else if (data.type === 'citations') {
        setMessages((m) => {
          const last = m[m.length - 1];
          if (last?.role === 'assistant') return [...m.slice(0, -1), { ...last, citations: data.items }];
          return [...m, { role: 'assistant', text: '', citations: data.items }];
        });
      } else if (data.type === 'stt') {
        setMessages((m) => [...m, { role: 'user', text: data.text }]);
      } else if (data.type === 'tts_audio') {
        playWav(data.data);
      }
    };
    ws.current = sock;
    return () => sock.close();
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  function send() {
    if (!draft.trim() || !ws.current) return;
    setMessages((m) => [...m, { role: 'user', text: draft }]);
    ws.current.send(JSON.stringify({ type: 'query', text: draft }));
    setDraft('');
  }

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    audioCtx.current = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.current.createMediaStreamSource(stream);
    const processor = audioCtx.current.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      const b64 = btoa(String.fromCharCode(...new Uint8Array(i16.buffer)));
      ws.current?.send(JSON.stringify({ type: 'audio_chunk', data: b64 }));
    };
    source.connect(processor);
    processor.connect(audioCtx.current.destination);
    recorder.current = { stream, processor, source } as unknown as MediaRecorder;
    setRecording(true);
  }

  function stopRecording() {
    const r = recorder.current as unknown as { stream: MediaStream; processor: ScriptProcessorNode };
    r?.stream.getTracks().forEach((t) => t.stop());
    r?.processor.disconnect();
    audioCtx.current?.close();
    setRecording(false);
    ws.current?.send(JSON.stringify({ type: 'audio_end' }));
  }

  function playWav(b64: string) {
    const audio = new Audio('data:audio/wav;base64,' + b64);
    audio.play().catch(() => {});
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 px-4 py-3 flex items-center justify-between">
        <h1 className="font-bold">RAG Voice Support Bot</h1>
        <span className={`text-xs px-2 py-1 rounded ${connected ? 'bg-green-700' : 'bg-red-700'}`}>
          {connected ? 'connected' : 'disconnected'}
        </span>
      </header>

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 py-6 overflow-auto">
        {messages.length === 0 && (
          <div className="text-center text-zinc-500 mt-20">
            <p className="mb-3">Ask me anything about the knowledge base.</p>
            <p className="text-sm">Try: <em>"How do I create a payment intent?"</em></p>
          </div>
        )}

        <div className="space-y-4">
          {messages.map((m, i) => (
            <div key={i} className={`p-4 rounded-lg ${m.role === 'user' ? 'bg-blue-900/40 ml-12' : 'bg-zinc-900 mr-12'}`}>
              <div className="text-xs uppercase tracking-wide text-zinc-400 mb-1">{m.role}</div>
              <div className="whitespace-pre-wrap">{m.text}</div>
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 space-y-1">
                  <div className="text-xs text-zinc-500">Sources:</div>
                  {m.citations.map((c) => (
                    <details key={c.chunk_id} className="text-xs bg-zinc-800 rounded p-2">
                      <summary className="cursor-pointer flex items-center gap-1">
                        <FileText className="w-3 h-3" /> [{c.index}] doc#{c.document_id} (score {c.score.toFixed(3)})
                      </summary>
                      <pre className="whitespace-pre-wrap mt-2 text-zinc-300">{c.text.slice(0, 500)}{c.text.length > 500 ? '...' : ''}</pre>
                    </details>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-zinc-800 px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center gap-2">
          <button
            onClick={recording ? stopRecording : startRecording}
            className={`p-2 rounded-md ${recording ? 'bg-red-600' : 'bg-zinc-800 hover:bg-zinc-700'}`}
            title={recording ? 'Stop recording' : 'Start recording'}
          >
            {recording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </button>
          <input
            className="flex-1 px-3 py-2 rounded-md bg-zinc-900 border border-zinc-700 focus:outline-none focus:border-blue-500"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && send()}
            placeholder={recording ? 'Listening...' : 'Type your question...'}
            disabled={recording}
          />
          <button onClick={send} className="p-2 rounded-md bg-blue-600 hover:bg-blue-500">
            <Send className="w-5 h-5" />
          </button>
        </div>
      </footer>
    </div>
  );
}
