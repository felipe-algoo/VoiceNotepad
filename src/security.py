import hashlib
import re
import pathlib
import threading
import time
import unicodedata
from typing import Final

ALLOWED_EXTENSIONS: Final[frozenset] = frozenset({".txt", ".md"})
MAX_FILENAME_LENGTH: Final[int] = 200
MAX_CONTENT_BYTES: Final[int] = 10 * 1024 * 1024
MAX_CONTENT_LINES: Final[int] = 500_000
MAX_TRANSCRIPTION_CHARS: Final[int] = 8192
MAX_SESSION_ID_LEN: Final[int] = 64

_UNSAFE_CHARS: Final = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_PATH_TRAVERSAL: Final = re.compile(r"(?:^|[\\/])\.\.(?:[\\/]|$)")
_NULL_BYTES: Final = re.compile(r"\x00")
_CONTROL_CHARS: Final = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SESSION_ID_RE: Final = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

_INJECTION_HIGH: Final = re.compile(
    r"(?i)(<\s*\|?\s*(?:im_start|im_end|system|assistant)\s*\|?\s*>|"
    r"\[/?INST\]|\[/?SYS\]|"
    r"###\s*(?:system|instruction)\b|"
    r"ignore\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules|context)|"
    r"disregard\s+(?:all\s+|the\s+|any\s+|your\s+)?(?:previous|prior|instructions?|training|guidelines|rules)|"
    r"you\s+are\s+now\s+(?:DAN|a\s+different\s+ai|an?\s+unrestricted)|"
    r"jailbreak\s+mode|dan\s+mode|developer\s+mode\s+enabled|"
    r"base64\s*:\s*[A-Za-z0-9+/=]{60,})"
)

_INJECTION_MEDIUM: Final = re.compile(
    r"(?i)(pretend\s+(?:to\s+be|you\s+are)|"
    r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:an?\s+)?(?:unethical|unrestricted|evil|hacker|jailbroken)|"
    r"system\s*:\s*you\s+(?:are|must|will|should)|"
    r"new\s+instructions?\s*:|"
    r"override\s+(?:your|all|the)\s+(?:instructions?|rules|guidelines)|"
    r"assistant\s*:\s*you\s+(?:must|will))"
)

_SENSITIVE_PATTERNS: Final = re.compile(
    r"(?i)(password\s*[:=]\s*\S+|api[_\s]?key\s*[:=]\s*\S+|secret\s*[:=]\s*\S+|"
    r"token\s*[:=]\s*[A-Za-z0-9\-._~+/]{8,}|"
    r"bearer\s+[A-Za-z0-9\-._~+/]+=*|"
    r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----|"
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b)"
)


class RateLimiter:
    def __init__(self, rate_per_sec: float, burst: int, max_keys: int = 4096) -> None:
        self._rate = float(rate_per_sec)
        self._burst = float(burst)
        self._max_keys = max_keys
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._buckets.clear()
                bucket = [self._burst, now]
                self._buckets[key] = bucket
            tokens = bucket[0] + (now - bucket[1]) * self._rate
            if tokens > self._burst:
                tokens = self._burst
            bucket[1] = now
            if tokens >= cost:
                bucket[0] = tokens - cost
                return True
            bucket[0] = tokens
            return False


def sanitize_filename(name: str) -> str:
    if not isinstance(name, str):
        return "untitled"
    name = unicodedata.normalize("NFC", name)
    name = _NULL_BYTES.sub("", name)
    name = _UNSAFE_CHARS.sub("_", name)
    name = name.strip(". ")
    if not name:
        return "untitled"
    return name[:MAX_FILENAME_LENGTH]


def validate_path(raw) -> pathlib.Path:
    if isinstance(raw, str) and _NULL_BYTES.search(raw):
        raise ValueError("invalid path")
    path = pathlib.Path(raw)
    if _PATH_TRAVERSAL.search(str(path)):
        raise ValueError("path traversal detected")
    safe_name = sanitize_filename(path.name)
    suffix = pathlib.Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("extension not permitted")
    resolved = (path.parent / safe_name).resolve()
    if _PATH_TRAVERSAL.search(str(resolved)):
        raise ValueError("path traversal detected")
    return resolved


def validate_content(content: str) -> None:
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    encoded = content.encode("utf-8", errors="replace")
    if len(encoded) > MAX_CONTENT_BYTES:
        raise ValueError("content too large")
    if content.count("\n") > MAX_CONTENT_LINES:
        raise ValueError("too many lines")


def sanitize_transcription(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _NULL_BYTES.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return text[:MAX_TRANSCRIPTION_CHARS]


def check_prompt_injection(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    if len(text) > MAX_TRANSCRIPTION_CHARS * 2:
        text = text[: MAX_TRANSCRIPTION_CHARS * 2]
    if _INJECTION_HIGH.search(text):
        return True
    medium_hits = _INJECTION_MEDIUM.findall(text)
    if len(medium_hits) >= 2:
        return True
    return False


def redact_sensitive(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _SENSITIVE_PATTERNS.sub("[REDACTED]", text)


def safe_error_message(exc: Exception) -> str:
    raw = str(exc) if exc is not None else ""
    raw = redact_sensitive(raw)
    raw = _CONTROL_CHARS.sub("", raw)
    if len(raw) > 200:
        raw = raw[:200]
    return raw or "internal error"


def content_fingerprint(content: str) -> str:
    data = content.encode("utf-8", errors="replace") if isinstance(content, str) else b""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def validate_session_id(session_id: str) -> bool:
    if not isinstance(session_id, str):
        return False
    if not session_id or len(session_id) > MAX_SESSION_ID_LEN:
        return False
    return bool(_SESSION_ID_RE.match(session_id))