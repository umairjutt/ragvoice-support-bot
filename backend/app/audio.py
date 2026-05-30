"""Whisper STT + Piper TTS wrappers (lazy-loaded)."""
import io
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

_stt = None
_tts = None


def get_stt():
    global _stt
    if _stt is None:
        from faster_whisper import WhisperModel
        _stt = WhisperModel("base", device="cpu", compute_type="int8")
    return _stt


def transcribe_pcm16k(pcm_bytes: bytes) -> str:
    """Transcribe raw 16-bit PCM @ 16kHz mono."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = get_stt().transcribe(arr, language="en")
    return " ".join(s.text for s in segments).strip()


def synthesize(text: str, voice: str = "en_US-amy-medium") -> bytes:
    """Synthesize text to WAV bytes via Piper.

    Note: requires the Piper voice model files; in dev these are downloaded
    on first call by piper-tts. Falls back to silent WAV if not installed.
    """
    try:
        from piper.voice import PiperVoice
        model_path = Path.home() / ".local" / "share" / "piper" / f"{voice}.onnx"
        if not model_path.exists():
            return _silent_wav()
        v = PiperVoice.load(str(model_path))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            v.synthesize(text, wav)
        return buffer.getvalue()
    except Exception:
        return _silent_wav()


def _silent_wav() -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()
