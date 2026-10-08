import logging
import queue
import threading
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from src.memory import EpisodicMemory
from src.security import sanitize_transcription, check_prompt_injection
from src.transcriber import Transcriber

logger = logging.getLogger(__name__)

try:
    import webrtcvad as _webrtcvad

    _WEBRTCVAD_AVAILABLE = True
except Exception:
    _WEBRTCVAD_AVAILABLE = False

_SAMPLE_RATE = 16000
_FRAME_MS = 30
_FRAME_SAMPLES = _SAMPLE_RATE * _FRAME_MS // 1000
_FRAME_BYTES = _FRAME_SAMPLES * 2
_PREROLL_FRAMES = 5
_PARTIAL_INTERVAL_MS = 900
_FINAL_SILENCE_MS = 700
_MIN_UTTERANCE_MS = 300
_MAX_UTTERANCE_MS = 45000
_VAD_AGGRESSIVENESS = 2
_MAX_WS_CHUNK_BYTES = 65536
_CONTEXT_WINDOW = 5


class _EnergyGate:
    _FLOOR = 3e-4
    _RATIO = 3.0

    def __init__(self) -> None:
        self._noise = self._FLOOR

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if arr.size == 0:
            return False
        energy = float(np.sqrt(np.mean(arr * arr)))
        threshold = max(self._FLOOR, self._noise * self._RATIO)
        speech = energy > threshold
        if not speech:
            self._noise = 0.98 * self._noise + 0.02 * energy
        return speech


class StreamingSession:
    def __init__(
        self,
        session_id: str,
        language: str,
        transcriber: Transcriber,
        memory: EpisodicMemory,
        sender: Callable[[dict[str, Any]], None],
    ) -> None:
        self.session_id = session_id
        self.language = language
        self._transcriber = transcriber
        self._memory = memory
        self._sender = sender

        if _WEBRTCVAD_AVAILABLE:
            self._vad = _webrtcvad.Vad(_VAD_AGGRESSIVENESS)
        else:
            self._vad = _EnergyGate()

        self._pcm_rem = b""
        self._preroll: deque[bytes] = deque(maxlen=_PREROLL_FRAMES)
        self._utt = bytearray()
        self._utt_ms = 0
        self._silence_ms = 0
        self._active = False
        self._utt_id = 0
        self._revision = 0
        self._last_partial_ms = 0
        self._partial_in_flight = False
        self._vad_last = False

        self._jobs: queue.Queue[Optional[tuple]] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._job_loop,
            daemon=True,
            name=f"stream-{session_id[:8]}",
        )
        self._worker.start()

    def feed(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if len(pcm) > _MAX_WS_CHUNK_BYTES:
            pcm = pcm[:_MAX_WS_CHUNK_BYTES]
        data = self._pcm_rem + pcm
        offset = 0
        n = len(data)
        while offset + _FRAME_BYTES <= n:
            frame = data[offset:offset + _FRAME_BYTES]
            offset += _FRAME_BYTES
            self._process_frame(frame)
        self._pcm_rem = data[offset:]

    def _process_frame(self, frame: bytes) -> None:
        try:
            is_speech = bool(self._vad.is_speech(frame, _SAMPLE_RATE))
        except Exception:
            is_speech = False

        if is_speech != self._vad_last:
            self._vad_last = is_speech
            try:
                self._sender({"type": "vad", "active": is_speech})
            except Exception:
                pass

        if not self._active:
            self._preroll.append(frame)
            if is_speech:
                with self._lock:
                    self._active = True
                    self._utt = bytearray()
                    for f in self._preroll:
                        self._utt.extend(f)
                    self._utt_ms = len(self._preroll) * _FRAME_MS
                    self._silence_ms = 0
                    self._last_partial_ms = self._utt_ms
                    self._partial_in_flight = False
                    self._revision = 0
                    self._utt_id += 1
                    self._preroll.clear()
            return

        with self._lock:
            self._utt.extend(frame)
            self._utt_ms += _FRAME_MS
            if is_speech:
                self._silence_ms = 0
            else:
                self._silence_ms += _FRAME_MS

            should_finalize = (
                self._silence_ms >= _FINAL_SILENCE_MS
                or self._utt_ms >= _MAX_UTTERANCE_MS
            )
            should_partial = (
                not self._partial_in_flight
                and self._utt_ms - self._last_partial_ms >= _PARTIAL_INTERVAL_MS
                and self._utt_ms >= _MIN_UTTERANCE_MS
            )

        if should_finalize:
            self._finalize()
        elif should_partial:
            self._queue_partial()

    def _queue_partial(self) -> None:
        with self._lock:
            if self._partial_in_flight or not self._active:
                return
            self._partial_in_flight = True
            self._last_partial_ms = self._utt_ms
            self._revision += 1
            pcm = bytes(self._utt)
            utt_id = self._utt_id
            rev = self._revision
        try:
            self._jobs.put_nowait(("partial", pcm, utt_id, rev))
        except queue.Full:
            with self._lock:
                self._partial_in_flight = False

    def _finalize(self) -> None:
        with self._lock:
            pcm = bytes(self._utt)
            utt_ms = self._utt_ms
            utt_id = self._utt_id
            self._active = False
            self._utt = bytearray()
            self._utt_ms = 0
            self._silence_ms = 0
            self._preroll.clear()
            self._partial_in_flight = False
            self._revision = 0
        if utt_ms >= _MIN_UTTERANCE_MS and pcm:
            try:
                self._jobs.put_nowait(("final", pcm, utt_id, 0))
            except queue.Full:
                pass

    def _job_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            kind, pcm, utt_id, rev = job
            try:
                self._run_job(kind, pcm, utt_id, rev)
            except Exception:
                logger.exception("streaming job failed")
            finally:
                if kind == "partial":
                    with self._lock:
                        self._partial_in_flight = False

    def _run_job(self, kind: str, pcm: bytes, utt_id: int, rev: int) -> None:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return

        context = None
        if kind == "final":
            try:
                context = self._memory.recent_context(self.session_id, limit=_CONTEXT_WINDOW)
            except Exception:
                context = None

        text, conf, _ = self._transcriber.transcribe(
            audio,
            partial=(kind == "partial"),
            context=context,
            language=self.language,
        )
        if not text:
            if kind == "final":
                self._safe_send({
                    "type": "final",
                    "utt_id": utt_id,
                    "text": "",
                    "confidence": 0.0,
                })
            return

        text = sanitize_transcription(text)
        if not text:
            return

        if check_prompt_injection(text):
            try:
                self._memory.record_event(self.session_id, "injection_filtered", text[:256])
            except Exception:
                pass
            self._safe_send({"type": "warning", "utt_id": utt_id, "message": "content_filtered"})
            return

        if kind == "partial":
            self._safe_send({
                "type": "partial",
                "utt_id": utt_id,
                "revision": rev,
                "text": text,
                "confidence": float(conf),
            })
        else:
            try:
                self._memory.add_segment(self.session_id, text, float(conf), is_final=True)
            except Exception:
                logger.exception("memory write failed")
            self._safe_send({
                "type": "final",
                "utt_id": utt_id,
                "text": text,
                "confidence": float(conf),
            })

    def _safe_send(self, msg: dict[str, Any]) -> None:
        try:
            self._sender(msg)
        except Exception:
            logger.debug("sender failed", exc_info=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._finalize()
        except Exception:
            logger.exception("finalize on close failed")
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=8.0)