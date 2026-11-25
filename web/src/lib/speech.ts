/** Match the spoken language as well as the locale (both Chinese options start with zh). */
function pickVoice(lang: string): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  const normalize = (voice: SpeechSynthesisVoice) => voice.lang.replace(/_/g, '-').toLowerCase();
  const compatible: Record<string, RegExp> = {
    'zh-HK': /^(yue(?:-|$)|zh-(?:hant-)?hk$)/,
    'zh-CN': /^(cmn(?:-|$)|zh-(?:hans(?:-|$)|cn$|tw$|sg$))/,
    'en-US': /^en(?:-|$)/,
  };
  for (const pool of [voices.filter((voice) => voice.localService), voices]) {
    const match = pool.find((voice) => normalize(voice) === lang.toLowerCase()) ||
      pool.find((voice) => compatible[lang]?.test(normalize(voice)));
    if (match) return match;
  }
}

let current: SpeechSynthesisUtterance | null = null;

export function cancelSpeech(): void {
  current = null;
  window.speechSynthesis?.cancel();
}

export function speak(text: string, lang: string, onEnd: (error?: string) => void): boolean {
  cancelSpeech();
  if (!text || !window.speechSynthesis) return false;
  const voice = pickVoice(lang);
  if (!voice) {
    const language = ({ 'zh-HK': 'Cantonese', 'zh-CN': 'Mandarin', 'en-US': 'English' })[lang as 'zh-HK' | 'zh-CN' | 'en-US'] || lang;
    onEnd('No ' + language + ' voice is available. Enable a matching system voice and select Replay.');
    return false;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = lang;
  current = utterance;
  function finish(error?: string) {
    if (current !== utterance) return;
    current = null;
    onEnd(error);
  }
  utterance.onend = () => finish();
  utterance.onerror = () => finish('Playback stopped. Select Replay to try again.');
  try {
    window.speechSynthesis.speak(utterance);
    return true;
  } catch {
    finish('Playback stopped. Select Replay to try again.');
    return false;
  }
}
