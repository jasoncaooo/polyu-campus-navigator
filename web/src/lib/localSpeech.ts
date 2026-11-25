/* Captures one question for the local transcription endpoint. */

function encodeSpeechWav(chunks: Float32Array[], sampleCount: number, inputRate: number) {
  const rate = 16000;
  const length = Math.floor(sampleCount * rate / inputRate);
  const bytes = new ArrayBuffer(44 + length * 2);
  const view = new DataView(bytes);
  function writeTag(offset: number, tag: string) {
    for (let i = 0; i < tag.length; i++) view.setUint8(offset + i, tag.charCodeAt(i));
  }
  writeTag(0, 'RIFF');
  view.setUint32(4, bytes.byteLength - 8, true);
  writeTag(8, 'WAVE');
  writeTag(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeTag(36, 'data');
  view.setUint32(40, length * 2, true);

  const samples = new Float32Array(sampleCount);
  let offset = 0;
  for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length; }
  // Average source samples in each output interval before quantizing to PCM16.
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * inputRate / rate);
    const end = Math.max(start + 1, Math.floor((i + 1) * inputRate / rate));
    let sum = 0;
    for (let j = start; j < end; j++) sum += samples[j];
    const value = Math.max(-1, Math.min(1, sum / (end - start)));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return bytes;
}

export interface LocalInputEvents {
  listening(): void;
  transcribing(): void;
  result(text: string): void;
  error(code: string, detail?: string): void;
  end(): void;
}

export class LocalSpeechInput {
  private chunks: Float32Array[] = [];
  private sampleCount = 0;
  private lastSpeech = 0;
  private hasSignal = false;
  private phase: 'opening' | 'recording' | 'transcribing' | 'ended' = 'opening';
  private request = new AbortController();
  private rate = 0;
  private limit?: number;
  private context?: AudioContext;
  private stream?: MediaStream;
  private capture?: AudioWorkletNode;

  static supported() {
    return Boolean(window.AudioContext && window.AudioWorkletNode && navigator.mediaDevices?.getUserMedia);
  }

  constructor(private lang: string, private events: LocalInputEvents) {}

  async start() {
    try {
      const context = new AudioContext();
      this.context = context;
      this.rate = context.sampleRate;
      // Resume on the microphone click, while it still counts as a user gesture.
      const ready = Promise.all([context.resume(), context.audioWorklet.addModule(import.meta.env.BASE_URL + 'mic-worklet.js')]);
      const permission = navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      }).then((stream) => {
        if (this.phase === 'opening') this.stream = stream;
        else stream.getTracks().forEach((track) => track.stop());
      });
      await Promise.all([permission, ready]);
      if (this.phase !== 'opening' || !this.stream) return;
      this.capture = new AudioWorkletNode(context, 'microphone-capture', {
        channelCount: 1, channelCountMode: 'explicit', outputChannelCount: [1],
      });
      const source = context.createMediaStreamSource(this.stream);
      source.connect(this.capture).connect(context.destination);
      this.capture.port.onmessage = (event) => this.collect(event.data);
      this.capture.onprocessorerror = () => this.fail('audio-capture');
      this.stream.getAudioTracks()[0].onended = () => this.fail('audio-capture');
      this.phase = 'recording';
      this.limit = window.setTimeout(() => this.stop(), 30000);
      this.events.listening();
    } catch (err) {
      if (this.phase !== 'opening') return;
      const code = err instanceof DOMException && err.name === 'NotAllowedError' ? 'not-allowed' : 'audio-capture';
      this.fail(code);
    }
  }

  private collect(samples: Float32Array) {
    if (this.phase !== 'recording') return;
    const chunk = samples.subarray(0, this.rate * 30 - this.sampleCount);
    this.chunks.push(chunk);
    this.sampleCount += chunk.length;
    let power = 0;
    for (const value of chunk) power += value * value;
    if (power > chunk.length * 0.0005 ** 2) this.hasSignal = true;
    if (power > chunk.length * 0.008 ** 2) this.lastSpeech = this.sampleCount;
    const silence = this.sampleCount - this.lastSpeech;
    if (!this.hasSignal && this.sampleCount >= this.rate * 8) {
      this.fail('no-speech');
      return;
    }
    if (this.sampleCount === this.rate * 30 ||
        (this.lastSpeech && silence >= this.rate * 1.5)) this.stop();
  }

  private releaseMic() {
    clearTimeout(this.limit);
    if (this.capture) {
      this.capture.port.onmessage = null;
      this.capture.disconnect();
      this.capture = undefined;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => { track.onended = null; track.stop(); });
      this.stream = undefined;
    }
    if (this.context) {
      this.context.close().catch(() => {});
      this.context = undefined;
    }
  }

  abort() {
    this.phase = 'ended';
    this.request.abort();
    this.releaseMic();
    this.chunks.length = 0;
  }

  private fail(error: string, detail?: string) {
    if (this.phase === 'ended') return;
    this.abort();
    this.events.error(error, detail);
    this.events.end();
  }

  async stop() {
    if (this.phase !== 'recording') return;
    this.phase = 'transcribing';
    this.releaseMic();
    if (!this.hasSignal) {
      this.fail('no-speech');
      return;
    }
    this.events.transcribing();
    try {
      const body = encodeSpeechWav(this.chunks, this.sampleCount, this.rate);
      this.chunks.length = 0;
      const res = await fetch('/api/transcribe?lang=' + encodeURIComponent(this.lang), {
        method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body, signal: this.request.signal,
      });
      const reply = await res.json();
      if (this.phase !== 'transcribing') return;
      if (!res.ok || reply.error) {
        this.fail('local-transcription', reply.error);
        return;
      }
      if (reply.text?.trim()) this.events.result(reply.text.trim());
      else { this.fail('no-speech'); return; }
      this.phase = 'ended';
      this.events.end();
    } catch {
      this.fail('local-transcription');
    }
  }
}
