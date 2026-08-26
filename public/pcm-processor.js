class PCM16Downsampler extends AudioWorkletProcessor {
  constructor() { super(); this.pending = []; this.phase = 0; this.sum = 0; this.count = 0; }
  process(inputs) {
    const input = inputs[0]?.[0]; if (!input) return true;
    const ratio = sampleRate / 16000; let energy = 0;
    for (let i = 0; i < input.length; i++) {
      const sample = input[i]; energy += sample * sample; this.sum += sample; this.count++; this.phase++;
      if (this.phase >= ratio) { const average = Math.max(-1, Math.min(1, this.sum / this.count)); this.pending.push(average < 0 ? average * 32768 : average * 32767); this.phase -= ratio; this.sum = 0; this.count = 0; }
    }
    if (this.pending.length >= 1600) { const pcm = new Int16Array(this.pending.splice(0, 1600)); const rms = Math.sqrt(energy / input.length); this.port.postMessage({ pcm: pcm.buffer, rms }, [pcm.buffer]); }
    return true;
  }
}
registerProcessor('pcm16-downsampler', PCM16Downsampler);
