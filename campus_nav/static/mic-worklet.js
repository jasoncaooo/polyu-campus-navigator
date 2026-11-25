// Batch microphone samples on the audio thread. The output stays silent.
class MicrophoneCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(1024);
    this.used = 0;
  }

  process(inputs) {
    const mono = inputs[0][0];
    if (!mono) return true;
    let offset = 0;
    while (offset < mono.length) {
      const count = Math.min(this.buffer.length - this.used, mono.length - offset);
      this.buffer.set(mono.subarray(offset, offset + count), this.used);
      this.used += count;
      offset += count;
      if (this.used === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(1024);
        this.used = 0;
      }
    }
    return true;
  }
}

registerProcessor('microphone-capture', MicrophoneCapture);
