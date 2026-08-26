'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

type Status = 'connecting' | 'ready' | 'recording' | 'thinking' | 'speaking' | 'error';
type Metrics = { speechDuration?: number; firstPartial?: number; sttFinal?: number; ttft?: number; tokensPerSecond?: number; llmTotal?: number; ttsFirstAudio?: number; ttsRealtime?: number; total?: number };
const emptyMetrics: Metrics = {};

function Metric({ label, value, unit = 'ms' }: { label: string; value?: number; unit?: string }) {
  const decimal = unit === 'tok/s' || unit === '× realtime';
  return <div className="metric"><span>{label}</span><strong>{value === undefined ? '—' : value.toFixed(decimal ? 1 : 0)} <small>{value === undefined ? '' : unit}</small></strong></div>;
}

export default function Home() {
  const [status, setStatus] = useState<Status>('connecting');
  const [transcript, setTranscript] = useState('Press and hold the button, then start speaking.');
  const [answer, setAnswer] = useState('');
  const [metrics, setMetrics] = useState<Metrics>(emptyMetrics);
  const [error, setError] = useState('');
  const [level, setLevel] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const recordingRef = useRef(false);
  const audioRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const playbackAtRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const speechEndRef = useRef(0);
  const receivedFirstAudioRef = useRef(false);

  const playAudio = useCallback(async (data: ArrayBuffer) => {
    const context = audioRef.current;
    if (!context) return;
    try {
      const buffer = await context.decodeAudioData(data.slice(0));
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      const at = Math.max(context.currentTime + 0.025, playbackAtRef.current);
      source.start(at);
      if (!receivedFirstAudioRef.current && speechEndRef.current) {
        receivedFirstAudioRef.current = true;
        const decodeAndNetworkMs = performance.now() - speechEndRef.current;
        const scheduledDelayMs = Math.max(0, at - context.currentTime) * 1000;
        setMetrics((m) => ({ ...m, total: decodeAndNetworkMs + scheduledDelayMs }));
      }
      playbackAtRef.current = at + buffer.duration;
      setStatus('speaking');
      source.onended = () => { if (context.currentTime >= playbackAtRef.current - 0.05) setStatus('ready'); };
    } catch { setError('The browser could not decode the returned audio.'); setStatus('error'); }
  }, []);

  useEffect(() => {
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${scheme}://${window.location.host}/ws/voice`);
      socket.binaryType = 'arraybuffer';
      socketRef.current = socket;
      socket.onopen = () => { setStatus('ready'); setError(''); };
      socket.onclose = () => { if (socketRef.current === socket) { setStatus('connecting'); retry = setTimeout(connect, 1500); } };
      socket.onerror = () => setError('Cannot reach the voice backend. Reconnecting…');
      socket.onmessage = (message) => {
        if (message.data instanceof ArrayBuffer) { void playAudio(message.data); return; }
        const event = JSON.parse(message.data);
        switch (event.type) {
          case 'stt_partial': setTranscript(event.text || 'Listening…'); setMetrics((m) => ({ ...m, firstPartial: m.firstPartial ?? event.first_partial_ms })); break;
          case 'stt_final': setTranscript(event.text || 'No speech detected'); setMetrics((m) => ({ ...m, speechDuration: event.speech_duration_ms, sttFinal: event.stt_final_ms })); setStatus('thinking'); break;
          case 'llm_token': setAnswer((text) => text + event.delta); setMetrics((m) => ({ ...m, ttft: m.ttft ?? event.ttft_ms })); break;
          case 'llm_done': setMetrics((m) => ({ ...m, tokensPerSecond: event.tokens_per_second, llmTotal: event.total_ms })); break;
          case 'tts_audio': setMetrics((m) => ({ ...m, ttsFirstAudio: m.ttsFirstAudio ?? event.ttfa_ms, ttsRealtime: event.realtime_factor })); break;
          case 'turn_done': setStatus((s) => s === 'speaking' ? s : 'ready'); break;
          case 'error': setError(event.message); setStatus('error'); break;
        }
      };
    };
    connect();
    return () => { clearTimeout(retry); socketRef.current?.close(); socketRef.current = null; };
  }, [playAudio]);

  const prepareMicrophone = useCallback(async () => {
    if (workletRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    streamRef.current = stream;
    const context = new AudioContext();
    audioRef.current = context;
    await context.audioWorklet.addModule('/pcm-processor.js');
    const source = context.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(context, 'pcm16-downsampler');
    const silent = context.createGain(); silent.gain.value = 0;
    source.connect(worklet).connect(silent).connect(context.destination);
    worklet.port.onmessage = (event) => {
      const { pcm, rms } = event.data; setLevel(Math.min(1, rms * 7));
      const socket = socketRef.current;
      if (recordingRef.current && socket?.readyState === WebSocket.OPEN) socket.send(pcm);
    };
    workletRef.current = worklet;
  }, []);

  const startRecording = useCallback(async () => {
    if (recordingRef.current || status === 'connecting') return;
    try {
      await prepareMicrophone(); await audioRef.current?.resume(); playbackAtRef.current = audioRef.current?.currentTime ?? 0;
      setMetrics(emptyMetrics); setTranscript('Listening…'); setAnswer(''); setError(''); receivedFirstAudioRef.current = false; speechEndRef.current = 0; recordingRef.current = true; setStatus('recording');
      socketRef.current?.send(JSON.stringify({ type: 'start', client_time_ms: performance.timeOrigin + performance.now(), sample_rate: 16000 }));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Microphone permission was denied.'); setStatus('error'); }
  }, [prepareMicrophone, status]);

  const stopRecording = useCallback(() => {
    if (!recordingRef.current) return;
    recordingRef.current = false; speechEndRef.current = performance.now(); setLevel(0); setStatus('thinking');
    socketRef.current?.send(JSON.stringify({ type: 'stop', client_time_ms: performance.timeOrigin + performance.now() }));
  }, []);

  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.code === 'Space' && !e.repeat && !(e.target instanceof HTMLInputElement)) { e.preventDefault(); void startRecording(); } };
    const up = (e: KeyboardEvent) => { if (e.code === 'Space') { e.preventDefault(); stopRecording(); } };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); streamRef.current?.getTracks().forEach((track) => track.stop()); void audioRef.current?.close(); };
  }, [startRecording, stopRecording]);

  const statusText: Record<Status, string> = { connecting: 'Connecting', ready: 'Ready', recording: 'Listening', thinking: 'Thinking', speaking: 'Speaking', error: 'Needs attention' };
  return (
    <main className="shell">
      <header className="topbar"><div className="brand"><span className="brand-mark">V</span><div><strong>VoiceBench</strong><small>LOCAL LATENCY LAB</small></div></div><div className={`connection ${status}`}><i />{statusText[status]}</div></header>
      <section className="hero"><div><p className="eyebrow">END-TO-END VOICE PIPELINE</p><h1>Hear exactly where<br />the milliseconds go.</h1></div><p className="intro">A local, private benchmark for Moonshine, Gemma, and Piper. Audio never leaves your server.</p></section>
      <section className="workspace">
        <div className="conversation">
          <div className="pipeline" aria-label="Active model pipeline">
            <div><span className="step">01</span><p>Speech to text<strong>Moonshine Tiny Streaming</strong></p></div><b>→</b>
            <div><span className="step">02</span><p>Language model<strong>Gemma 3 1B · Q4_K_M</strong></p></div><b>→</b>
            <div><span className="step">03</span><p>Text to speech<strong>Piper · Lessac Medium</strong></p></div>
          </div>
          <div className="turn"><div className="speaker"><span>YOU</span><i /></div><p className={status === 'recording' ? 'live-text' : ''}>{transcript}</p></div>
          <div className="turn answer"><div className="speaker"><span>AI</span><i /></div><p>{answer || (status === 'thinking' ? 'Generating a response…' : 'The model response will appear here.')}</p></div>
          <div className="talk-zone"><div className="wave" aria-hidden="true">{Array.from({ length: 28 }, (_, i) => <i key={i} style={{ height: `${8 + Math.sin(i * 1.8) * 5 + level * (12 + (i % 5) * 6)}px` }} />)}</div>
            <button className={`talk-button ${status === 'recording' ? 'active' : ''}`} onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); void startRecording(); }} onPointerUp={stopRecording} onPointerCancel={stopRecording} disabled={status === 'connecting'} aria-label="Hold to talk"><span>{status === 'recording' ? '■' : '●'}</span>{status === 'recording' ? 'Release to send' : 'Hold to talk'}</button><small>or hold the space bar</small></div>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </div>
        <aside className="dashboard"><div className="dashboard-head"><div><p className="eyebrow">LIVE MEASUREMENTS</p><h2>Latency</h2></div><span>THIS TURN</span></div>
          <div className="total"><span>End speech → first audio</span><strong>{metrics.total === undefined ? '—' : metrics.total.toFixed(0)}<small>{metrics.total === undefined ? '' : ' ms'}</small></strong><div><i style={{ width: `${Math.min(100, (metrics.total ?? 0) / 10)}%` }} /></div></div>
          <div className="metric-group"><h3><span>STT</span>Moonshine</h3><Metric label="Speech duration" value={metrics.speechDuration} /><Metric label="First partial" value={metrics.firstPartial} /><Metric label="Final after end" value={metrics.sttFinal} /></div>
          <div className="metric-group"><h3><span>LLM</span>Gemma 3</h3><Metric label="Time to first token" value={metrics.ttft} /><Metric label="Generation speed" value={metrics.tokensPerSecond} unit="tok/s" /><Metric label="Total generation" value={metrics.llmTotal} /></div>
          <div className="metric-group"><h3><span>TTS</span>Piper</h3><Metric label="Time to first audio" value={metrics.ttsFirstAudio} /><Metric label="Generation speed" value={metrics.ttsRealtime} unit="× realtime" /></div>
        </aside>
      </section>
      <footer><span>16 kHz PCM · WebSocket</span><span>All inference runs on your server</span></footer>
    </main>
  );
}
