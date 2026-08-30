'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

type Status = 'connecting' | 'ready' | 'recording' | 'thinking' | 'speaking' | 'error';
type RouteLanguage = 'auto' | 'en' | 'ar';
type Metrics = { speechDuration?: number; firstPartial?: number; sttFinal?: number; ttft?: number; tokensPerSecond?: number; llmTotal?: number; ttsFirstAudio?: number; ttsRealtime?: number; total?: number };
const emptyMetrics: Metrics = {};

function Metric({ label, value, unit = 'ms' }: { label: string; value?: number; unit?: string }) {
  const decimal = unit === 'tok/s' || unit === '× realtime';
  return <div className="metric"><span>{label}</span><strong>{value === undefined ? '—' : value.toFixed(decimal ? 1 : 0)} <small>{value === undefined ? '' : unit}</small></strong></div>;
}

export default function Home() {
  const [status, setStatus] = useState<Status>('connecting');
  const [transcript, setTranscript] = useState('Start a call, then speak naturally.');
  const [answer, setAnswer] = useState('');
  const [metrics, setMetrics] = useState<Metrics>(emptyMetrics);
  const [error, setError] = useState('');
  const [level, setLevel] = useState(0);
  const [callActive, setCallActive] = useState(false);
  const [routeLanguage, setRouteLanguage] = useState<RouteLanguage>('auto');
  const [routeConfidence, setRouteConfidence] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const recordingRef = useRef(false);
  const callActiveRef = useRef(false);
  const statusRef = useRef<Status>('connecting');
  const preRollRef = useRef<ArrayBuffer[]>([]);
  const triggerFramesRef = useRef(0);
  const silenceFramesRef = useRef(0);
  const voicedFramesRef = useRef(0);
  const noiseFloorRef = useRef(0.004);
  const listenAfterRef = useRef(0);
  const audioRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const playbackAtRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const speechEndRef = useRef(0);
  const receivedFirstAudioRef = useRef(false);

  useEffect(() => { statusRef.current = status; }, [status]);

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
      statusRef.current = 'speaking';
      setStatus('speaking');
      source.onended = () => {
        if (context.currentTime >= playbackAtRef.current - 0.05) {
          listenAfterRef.current = performance.now() + 500;
          statusRef.current = 'ready';
          setStatus('ready');
        }
      };
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
      socket.onclose = () => { if (socketRef.current === socket) { callActiveRef.current = false; setCallActive(false); setStatus('connecting'); retry = setTimeout(connect, 1500); } };
      socket.onerror = () => setError('Cannot reach the voice backend. Reconnecting…');
      socket.onmessage = (message) => {
        if (message.data instanceof ArrayBuffer) { void playAudio(message.data); return; }
        const event = JSON.parse(message.data);
        switch (event.type) {
          case 'stt_partial': setTranscript(event.text || 'Listening…'); if (event.stable) { setRouteLanguage(event.language); setRouteConfidence(event.confidence || 0); } setMetrics((m) => ({ ...m, firstPartial: m.firstPartial ?? event.first_partial_ms })); break;
          case 'language_detected': setRouteLanguage(event.language); setRouteConfidence(event.confidence || 0); break;
          case 'stt_final': setTranscript(event.text || 'No speech detected'); setRouteLanguage(event.language || 'auto'); setRouteConfidence(event.confidence || 0); setMetrics((m) => ({ ...m, speechDuration: event.speech_duration_ms, sttFinal: event.stt_final_ms })); setStatus('thinking'); break;
          case 'stt_ignored': setTranscript('Listening for speech…'); setError(''); listenAfterRef.current = performance.now() + 350; setStatus('ready'); break;
          case 'llm_token': setAnswer((text) => text + event.delta); setMetrics((m) => ({ ...m, ttft: m.ttft ?? event.ttft_ms })); break;
          case 'llm_done': setMetrics((m) => ({ ...m, tokensPerSecond: event.tokens_per_second, llmTotal: event.total_ms })); break;
          case 'tts_audio': setMetrics((m) => ({ ...m, ttsFirstAudio: m.ttsFirstAudio ?? event.ttfa_ms, ttsRealtime: event.realtime_factor })); break;
          case 'turn_done': setStatus((s) => s === 'speaking' ? s : 'ready'); break;
          case 'error': setError(event.message); listenAfterRef.current = performance.now() + 600; setStatus(callActiveRef.current ? 'ready' : 'error'); break;
        }
      };
    };
    connect();
    return () => { clearTimeout(retry); socketRef.current?.close(); socketRef.current = null; };
  }, [playAudio]);

  const prepareMicrophone = useCallback(async () => {
    if (audioRef.current?.state === 'closed') {
      audioRef.current = null;
      workletRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (workletRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone access requires HTTPS when opening the app from a LAN IP. Use HTTPS or open it through an SSH tunnel at http://localhost:8080.');
    }
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
      const { pcm, rms }: { pcm: ArrayBuffer; rms: number } = event.data;
      setLevel(Math.min(1, rms * 7));
      const socket = socketRef.current;
      const canListen = callActiveRef.current && performance.now() >= listenAfterRef.current && (statusRef.current === 'ready' || statusRef.current === 'recording');
      if (!canListen || socket?.readyState !== WebSocket.OPEN) {
        preRollRef.current = [];
        triggerFramesRef.current = 0;
        return;
      }

      if (!recordingRef.current) {
        if (rms < Math.max(0.03, noiseFloorRef.current * 3)) {
          noiseFloorRef.current = noiseFloorRef.current * 0.94 + rms * 0.06;
        }
        preRollRef.current.push(pcm);
        if (preRollRef.current.length > 4) preRollRef.current.shift();
        const startThreshold = Math.max(0.016, noiseFloorRef.current * 3.2);
        triggerFramesRef.current = rms >= startThreshold ? triggerFramesRef.current + 1 : 0;
        if (triggerFramesRef.current < 3) return;

        recordingRef.current = true;
        triggerFramesRef.current = 0;
        silenceFramesRef.current = 0;
        voicedFramesRef.current = 0;
        receivedFirstAudioRef.current = false;
        speechEndRef.current = 0;
        playbackAtRef.current = audioRef.current?.currentTime ?? 0;
        setMetrics(emptyMetrics);
        setRouteLanguage('auto');
        setRouteConfidence(0);
        setTranscript('Listening…');
        setAnswer('');
        setError('');
        statusRef.current = 'recording';
        setStatus('recording');
        socket.send(JSON.stringify({ type: 'start', client_time_ms: performance.timeOrigin + performance.now(), sample_rate: 16000 }));
        for (const chunk of preRollRef.current) socket.send(chunk);
        preRollRef.current = [];
        return;
      }

      socket.send(pcm);
      const continueThreshold = Math.max(0.009, noiseFloorRef.current * 1.7);
      if (rms >= continueThreshold) {
        voicedFramesRef.current += 1;
        silenceFramesRef.current = 0;
      } else {
        silenceFramesRef.current += 1;
      }
      if (voicedFramesRef.current >= 2 && silenceFramesRef.current >= 8) {
        recordingRef.current = false;
        speechEndRef.current = performance.now();
        statusRef.current = 'thinking';
        setStatus('thinking');
        socket.send(JSON.stringify({ type: 'stop', client_time_ms: performance.timeOrigin + performance.now() }));
      }
    };
    workletRef.current = worklet;
  }, []);

  const toggleCall = useCallback(async () => {
    const socket = socketRef.current;
    if (callActiveRef.current) {
      callActiveRef.current = false;
      setCallActive(false);
      preRollRef.current = [];
      if (recordingRef.current) {
        recordingRef.current = false;
        speechEndRef.current = performance.now();
        socket?.send(JSON.stringify({ type: 'stop', client_time_ms: performance.timeOrigin + performance.now() }));
      }
      setLevel(0);
      statusRef.current = 'ready';
      setStatus('ready');
      setTranscript('Conversation ended. Start a new call when you are ready.');
      return;
    }
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setError('The voice backend is not connected yet.');
      return;
    }
    try {
      await prepareMicrophone();
      await audioRef.current?.resume();
      callActiveRef.current = true;
      noiseFloorRef.current = 0.004;
      setCallActive(true);
      listenAfterRef.current = performance.now() + 250;
      setTranscript('Call active — start speaking naturally.');
      setError('');
      statusRef.current = 'ready';
      setStatus('ready');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Microphone permission was denied.'); setStatus('error'); }
  }, [prepareMicrophone]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.code === 'Space' && !e.repeat && !(e.target instanceof HTMLInputElement)) { e.preventDefault(); void toggleCall(); } };
    window.addEventListener('keydown', down);
    return () => { window.removeEventListener('keydown', down); streamRef.current?.getTracks().forEach((track) => track.stop()); void audioRef.current?.close(); };
  }, [toggleCall]);

  const statusText: Record<Status, string> = { connecting: 'Connecting', ready: 'Ready', recording: 'Listening', thinking: 'Thinking', speaking: 'Speaking', error: 'Needs attention' };
  const routeLabel = routeLanguage === 'ar' ? 'العربية' : routeLanguage === 'en' ? 'English' : 'Auto detect';
  const llmLabel = routeLanguage === 'ar' ? 'RightNow Arabic 0.5B' : routeLanguage === 'en' ? 'Gemma 3 1B' : 'Gemma + RightNow';
  return (
    <main className="shell">
      <header className="topbar"><div className="brand"><span className="brand-mark">V</span><div><strong>VoiceBench</strong><small>LOCAL LATENCY LAB</small></div></div><div className={`connection ${status}`}><i />{callActive && status === 'ready' ? 'Listening' : statusText[status]}</div></header>
      <section className="hero"><div><p className="eyebrow">BILINGUAL END-TO-END VOICE PIPELINE</p><h1>Hear exactly where<br />the milliseconds go.</h1></div><p className="intro">A local English–Arabic benchmark with partial-STT language routing. Audio never leaves your server.</p></section>
      <section className="workspace">
        <div className="conversation">
          <div className="pipeline" aria-label="Active model pipeline">
            <div><span className="step">01</span><p>Speech + language<strong>Moonshine Tiny · EN + AR</strong></p></div><b>→</b>
            <div><span className="step">02</span><p>Language model<strong>{llmLabel}</strong></p></div><b>→</b>
            <div><span className="step">03</span><p>Text to speech<strong>Piper · Amy / Kareem</strong></p></div>
          </div>
          <div className="turn"><div className="speaker"><span>YOU</span><i /></div><p dir="auto" className={status === 'recording' ? 'live-text' : ''}>{transcript}</p></div>
          <div className="turn answer"><div className="speaker"><span>AI</span><i /></div><p dir="auto">{answer || (status === 'thinking' ? 'Generating a response…' : 'The model response will appear here.')}</p></div>
          <div className="talk-zone"><div className="wave" aria-hidden="true">{Array.from({ length: 28 }, (_, i) => <i key={i} style={{ height: `${8 + Math.sin(i * 1.8) * 5 + level * (12 + (i % 5) * 6)}px` }} />)}</div>
            <button className={`talk-button ${callActive ? 'active' : ''}`} onClick={() => void toggleCall()} disabled={status === 'connecting'} aria-label={callActive ? 'End conversation' : 'Start conversation'}><span>{callActive ? '■' : '●'}</span>{callActive ? 'End call' : 'Start call'}</button><small>{callActive ? 'Voice detection is active' : 'or press the space bar'}</small></div>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </div>
        <aside className="dashboard"><div className="dashboard-head"><div><p className="eyebrow">LIVE MEASUREMENTS</p><h2>Latency</h2></div><span className={`route-badge ${routeLanguage}`}>{routeLabel}{routeConfidence > 0 ? ` · ${Math.round(routeConfidence * 100)}%` : ''}</span></div>
          <div className="total"><span>End speech → first audio</span><strong>{metrics.total === undefined ? '—' : metrics.total.toFixed(0)}<small>{metrics.total === undefined ? '' : ' ms'}</small></strong><div><i style={{ width: `${Math.min(100, (metrics.total ?? 0) / 10)}%` }} /></div></div>
          <div className="metric-group"><h3><span>STT</span>Moonshine</h3><Metric label="Speech duration" value={metrics.speechDuration} /><Metric label="First partial" value={metrics.firstPartial} /><Metric label="Final after end" value={metrics.sttFinal} /></div>
          <div className="metric-group"><h3><span>LLM</span>{llmLabel}</h3><Metric label="Time to first token" value={metrics.ttft} /><Metric label="Generation speed" value={metrics.tokensPerSecond} unit="tok/s" /><Metric label="Total generation" value={metrics.llmTotal} /></div>
          <div className="metric-group"><h3><span>TTS</span>Piper</h3><Metric label="Time to first audio" value={metrics.ttsFirstAudio} /><Metric label="Generation speed" value={metrics.ttsRealtime} unit="× realtime" /></div>
        </aside>
      </section>
      <footer><span>16 kHz PCM · WebSocket</span><span>All inference runs on your server</span></footer>
    </main>
  );
}
