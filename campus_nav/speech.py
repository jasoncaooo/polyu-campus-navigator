"""Transcribe short voice drafts with a local whisper.cpp executable."""
from __future__ import annotations

import io
import os
import subprocess
import threading
import wave
from pathlib import Path

from .config import PROJECT_ROOT

SAMPLE_RATE = 16_000
MAX_SECONDS = 30
MAX_WAV_BYTES = 2 * SAMPLE_RATE * MAX_SECONDS + 4096  # allow space for WAV metadata
LANGUAGES = {"en-US": "en", "zh-HK": "yue", "zh-CN": "zh"}
DEFAULT_MODEL = ".voice/models/ggml-large-v3-turbo-q5_0.bin"
# Place names give the decoder some spelling context for mixed-language questions.
CAMPUS_NAMES = ("Core A, Core D, Core P, Block N, Block V, Block X, Block Y, Block Z, "
                "Innovation Tower, Pao Yue-kong Library.")
PROMPT_INTROS = {
    "en-US": "PolyU campus places: ",
    "zh-HK": "理大校園地點：",
    "zh-CN": "理大校园地点：",
}


class SpeechError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _pcm_wav(audio: bytes) -> bytes:
    """Check the declared format and length, then discard all non-audio chunks."""
    if len(audio) > MAX_WAV_BYTES:
        raise SpeechError(413, "Record up to 30 seconds at a time.")
    try:
        with wave.open(io.BytesIO(audio)) as incoming:
            fmt = (incoming.getframerate(), incoming.getnchannels(),
                   incoming.getsampwidth(), incoming.getcomptype())
            if fmt != (SAMPLE_RATE, 1, 2, "NONE"):
                raise SpeechError(415, "Record a mono, 16 kHz, 16-bit PCM WAV file.")
            count = incoming.getnframes()
            if count > SAMPLE_RATE * MAX_SECONDS:
                raise SpeechError(413, "Record up to 30 seconds at a time.")
            pcm = incoming.readframes(count)
            if len(pcm) != count * 2:
                raise SpeechError(400, "The recording is incomplete. Please record again.")
    except (EOFError, OSError, RuntimeError, ValueError, wave.Error) as exc:
        raise SpeechError(400, "The recording is not a readable WAV file.") from exc
    if not any(pcm):
        raise SpeechError(422, "No speech was detected. Check the microphone and try again.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, SAMPLE_RATE, count, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return output.getvalue()


class LocalTranscriber:
    def __init__(self):
        binary_name = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
        cli = os.environ.get("WHISPER_CLI") or f".voice/bin/{binary_name}"
        model = os.environ.get("WHISPER_MODEL") or DEFAULT_MODEL
        self.cli = PROJECT_ROOT / Path(cli).expanduser()
        self.model = PROJECT_ROOT / Path(model).expanduser()
        self._lock = threading.Lock()

    def status(self) -> dict:
        error = ""
        if not self.cli.is_file() or not os.access(self.cli, os.X_OK):
            error = "The local Whisper executable is missing or cannot be run."
        elif not self.model.is_file() or not os.access(self.model, os.R_OK):
            error = "The local Whisper speech model is missing or cannot be read."
        return {"available": not error, "max_seconds": MAX_SECONDS, "error": error}

    def transcribe(self, audio: bytes, lang: str) -> str:
        if lang not in LANGUAGES:
            raise SpeechError(400, "Choose English, Cantonese, or Mandarin.")
        language = LANGUAGES[lang]
        # Whisper Small predates the dedicated Cantonese language token in large-v3.
        if language == "yue" and self.model.name.startswith("ggml-small"):
            language = "zh"
        wav = _pcm_wav(audio)
        error = self.status()["error"]
        if error:
            raise SpeechError(503, error)
        if not self._lock.acquire(blocking=False):
            raise SpeechError(429, "Another recording is being transcribed. Try again shortly.")
        try:
            # In pipe mode whisper-cli needs an explicit text output on stdout.
            # Both '-' paths keep recordings and transcripts in memory.
            result = subprocess.run(
                [str(self.cli), "-m", str(self.model), "-l", language,
                 "--prompt", PROMPT_INTROS[lang] + CAMPUS_NAMES,
                 "-t", "4", "-nt", "-np", "-otxt", "-of", "-", "-f", "-"],
                input=wav, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                check=True, timeout=90,
            )
        except subprocess.TimeoutExpired as exc:
            raise SpeechError(504, "Local transcription timed out. Try a shorter recording.") from exc
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SpeechError(503, "The local speech engine could not decode this recording.") from exc
        finally:
            self._lock.release()
        # whisper-cli puts each successive segment on its own line.
        text = " ".join(result.stdout.decode("utf-8", errors="replace").split())
        if not text or text.casefold() in ("[blank_audio]", "[silence]"):
            raise SpeechError(422, "No speech was detected. Please record again.")
        return text
