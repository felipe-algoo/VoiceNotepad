import logging
import threading
from typing import Optional

import numpy as np

from src.security import sanitize_transcription, check_prompt_injection

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 5
_CONFIDENCE_THRESHOLD = 0.35
_MIN_AUDIO_SECONDS = 0.25
_SAMPLE_RATE = 16_000
_SUPPORTED_LANGS = frozenset({"pt", "en", "auto"})
_MAX_CONTEXT_CHARS = 512
_MAX_AUDIO_SECONDS = 60.0
_PARTIAL_MAX_SECONDS = 12.0


class Transcriber:
    _DEFAULT_MODEL = "base"

    def __init__(self, language: str = "pt") -> None:
        self._lang = language if language in _SUPPORTED_LANGS else "pt"
        self._model = None
        self._lock = threading.Lock()

    @property
    def language(self) -> str:
        return self._lang

    @language.setter
    def language(self, value: str) -> None:
        if value in _SUPPORTED_LANGS:
            self._lang = value

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_size: str = _DEFAULT_MODEL) -> None:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=2,
            cpu_threads=4,
        )
        with self._lock:
            self._model = model
        logger.info("whisper model '%s' loaded", model_size)

    def transcribe(
        self,
        audio: np.ndarray,
        partial: bool = False,
        context: Optional[list[str]] = None,
        language: Optional[str] = None,
    ) -> tuple[str, float, str]:
        if self._model is None:
            raise RuntimeError("model not loaded")
        if audio is None or audio.size == 0:
            return "", 0.0, self._lang
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        min_samples = int(_MIN_AUDIO_SECONDS * _SAMPLE_RATE)
        if audio.size < min_samples:
            return "", 0.0, self._lang

        if partial:
            max_samples = int(_PARTIAL_MAX_SECONDS * _SAMPLE_RATE)
        else:
            max_samples = int(_MAX_AUDIO_SECONDS * _SAMPLE_RATE)
        if audio.size > max_samples:
            audio = audio[-max_samples:]

        if not np.isfinite(audio).all():
            return "", 0.0, self._lang

        prompt: Optional[str] = None
        if context and not partial:
            joined = " ".join(context[-_CONTEXT_WINDOW:])
            joined = sanitize_transcription(joined)
            if joined and not check_prompt_injection(joined):
                prompt = joined[:_MAX_CONTEXT_CHARS]

        lang_code = language if language in _SUPPORTED_LANGS else self._lang
        lang = lang_code if lang_code != "auto" else None

        if partial:
            beam_size = 1
            best_of = 1
            temperature = 0.0
            condition_prev = False
            vad_filter = False
        else:
            beam_size = 5
            best_of = 5
            temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            condition_prev = True
            vad_filter = True

        with self._lock:
            model = self._model
            segments, info = model.transcribe(
                audio,
                language=lang,
                initial_prompt=prompt,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=condition_prev,
                vad_filter=vad_filter,
            )

            parts: list[str] = []
            log_probs: list[float] = []
            threshold = 0.75 if partial else 0.6
            for seg in segments:
                if seg.no_speech_prob < threshold and seg.text.strip():
                    cleaned = sanitize_transcription(seg.text.strip())
                    if not cleaned:
                        continue
                    if check_prompt_injection(cleaned):
                        logger.warning("injection segment dropped")
                        continue
                    parts.append(cleaned)
                    log_probs.append(float(seg.avg_logprob))

            text = " ".join(parts).strip()
            confidence = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
            detected = getattr(info, "language", lang_code) or lang_code
            return text, confidence, detected

    def clear_context(self) -> None:
        return