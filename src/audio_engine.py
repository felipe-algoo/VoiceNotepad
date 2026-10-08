import collections
import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import webrtcvad as _webrtcvad

    _WEBRTCVAD_AVAILABLE = True
except Exception:
    _WEBRTCVAD_AVAILABLE = False
    logger.warning("webrtcvad unavailable – falling back to energy-based VAD.")


class _EnergyVAD:
    _ADAPT_RATE = 0.02
    _SNR_RATIO = 3.5
    _FLOOR = 4e-4
    _SPEECH_ONSET = 4
    _SILENCE_ONSET = 14

    def __init__(self) -> None:
        self._noise: float = self._FLOOR
        self._active: bool = False
        self._buf: list[np.ndarray] = []
        self._speech_n: int = 0
        self._silence_n: int = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._buf.clear()
        self._speech_n = 0
        self._silence_n = 0

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        energy = float(np.sqrt(np.mean(frame ** 2)))
        threshold = max(self._FLOOR, self._noise * self._SNR_RATIO)
        is_speech = energy > threshold

        if not self._active:
            self._noise += self._ADAPT_RATE * (energy - self._noise)
            if is_speech:
                self._speech_n += 1
                self._buf.append(frame)
                if self._speech_n >= self._SPEECH_ONSET:
                    self._active = True
                    self._silence_n = 0
            else:
                self._speech_n = max(0, self._speech_n - 1)
                if len(self._buf) > self._SPEECH_ONSET:
                    self._buf.pop(0)
        else:
            self._buf.append(frame)
            if is_speech:
                self._silence_n = 0
            else:
                self._silence_n += 1
                if self._silence_n >= self._SILENCE_ONSET:
                    audio = np.concatenate(self._buf)
                    self.reset()
                    return audio

        return None


class _WebRTCVAD:
    _SR = 16_000
    _FRAME_MS = 30
    _FRAME_SAMPLES = int(_SR * _FRAME_MS / 1_000)
    _FRAME_BYTES = _FRAME_SAMPLES * 2
    _PAD_FRAMES = 10
    _VOICED_RATIO = 0.85
    _UNVOICED_RATIO = 0.15

    def __init__(self, aggressiveness: int = 2) -> None:
        self._vad = _webrtcvad.Vad(aggressiveness)
        self._ring: collections.deque = collections.deque(maxlen=self._PAD_FRAMES)
        self._active: bool = False
        self._buf: list[bytes] = []
        self._pcm_remainder: bytes = b""

    @property
    def is_active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._buf.clear()
        self._ring.clear()
        self._pcm_remainder = b""

    def _iter_frames(self, pcm: bytes):
        data = self._pcm_remainder + pcm
        offset = 0
        while offset + self._FRAME_BYTES <= len(data):
            yield data[offset: offset + self._FRAME_BYTES]
            offset += self._FRAME_BYTES
        self._pcm_remainder = data[offset:]

    def process(self, pcm: bytes) -> Optional[np.ndarray]:
        result: Optional[np.ndarray] = None
        for frame in self._iter_frames(pcm):
            result = self._process_frame(frame)
            if result is not None:
                return result
        return None

    def _process_frame(self, frame: bytes) -> Optional[np.ndarray]:
        try:
            speech = self._vad.is_speech(frame, self._SR)
        except Exception:
            return None

        if not self._active:
            self._ring.append((frame, speech))
            n = len(self._ring)
            voiced = sum(1 for _, s in self._ring if s)
            if n > 0 and voiced / n >= self._VOICED_RATIO:
                self._active = True
                self._buf = [f for f, _ in self._ring]
                self._ring.clear()
        else:
            self._buf.append(frame)
            self._ring.append((frame, speech))
            n = len(self._ring)
            unvoiced = sum(1 for _, s in self._ring if not s)
            if n > 0 and unvoiced / n >= self._UNVOICED_RATIO:
                raw = b"".join(self._buf)
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0
                self.reset()
                return arr

        return None


class AudioEngine:
    SAMPLE_RATE: int = 16_000
    _CHANNELS: int = 1
    _BLOCKSIZE: int = 480
    _QUEUE_MAX: int = 300

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        on_vad_change: Callable[[bool], None],
        vad_aggressiveness: int = 2,
    ) -> None:
        self._on_utterance = on_utterance
        self._on_vad_change = on_vad_change
        self._running = False
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=self._QUEUE_MAX)
        self._last_vad = False

        if _WEBRTCVAD_AVAILABLE:
            self._vad: _WebRTCVAD | _EnergyVAD = _WebRTCVAD(vad_aggressiveness)
            self._use_webrtc = True
        else:
            self._vad = _EnergyVAD()
            self._use_webrtc = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        import sounddevice as sd
        self._vad.reset()
        self._last_vad = False
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self._CHANNELS,
            dtype=np.int16,
            blocksize=self._BLOCKSIZE,
            callback=self._sd_callback,
        )
        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="audio-vad",
        )
        self._thread.start()
        self._stream.start()
        logger.info("AudioEngine started (webrtcvad=%s)", self._use_webrtc)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.debug("Stream close error", exc_info=True)
            self._stream = None
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("AudioEngine stopped")

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            logger.debug("sounddevice status: %s", status)
        if self._running and not self._q.full():
            try:
                self._q.put_nowait(indata.copy())
            except queue.Full:
                pass

    def _process_loop(self) -> None:
        while self._running:
            try:
                frame = self._q.get(timeout=0.15)
            except queue.Empty:
                continue
            if frame is None:
                break

            if self._use_webrtc:
                result = self._vad.process(frame.tobytes())
            else:
                f32 = frame.flatten().astype(np.float32) / 32_768.0
                result = self._vad.process(f32)

            if result is not None and len(result) > 0:
                self._on_utterance(result)

            state = self._vad.is_active
            if state != self._last_vad:
                self._last_vad = state
                self._on_vad_change(state)
