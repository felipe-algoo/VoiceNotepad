import asyncio
import json
import logging
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.memory import EpisodicMemory
from src.security import RateLimiter, safe_error_message, validate_session_id
from src.streaming import StreamingSession
from src.transcriber import Transcriber

logger = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parent
_STATIC_DIR = _ROOT / "static"
_DB_PATH = _ROOT.parent / "voice_notepad.db"

_ALLOWED_LANGS = frozenset({"pt", "en", "auto"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    app.state.loop = loop
    app.state.memory = EpisodicMemory(_DB_PATH)
    app.state.transcriber = Transcriber(language="pt")
    app.state.http_limiter = RateLimiter(rate_per_sec=30.0, burst=90)
    app.state.ws_limiter = RateLimiter(rate_per_sec=8.0, burst=24)
    app.state.msg_limiter = RateLimiter(rate_per_sec=200.0, burst=600)
    app.state.model_ready = False

    try:
        await loop.run_in_executor(None, app.state.transcriber.load, "base")
        app.state.model_ready = True
    except Exception:
        logger.exception("model load failed")

    yield

    try:
        app.state.memory.close()
    except Exception:
        pass


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = 'microphone=(self "http://localhost:" "http://127.0.0.1:"), '
    'camera=(), geolocation=()'
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "connect-src 'self' ws: wss:; "
        "img-src 'self' data:; "
        "media-src 'self' blob:; "
        "worker-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self';"
    )
    return response


def _client_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = _STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>UI not installed</h1>", status_code=500)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_ready": bool(app.state.model_ready),
    })


@app.get("/api/sessions")
async def list_sessions(request: Request, limit: int = 50) -> JSONResponse:
    if not app.state.http_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    limit = max(1, min(int(limit), 200))
    return JSONResponse({"sessions": app.state.memory.list_sessions(limit=limit)})


@app.post("/api/sessions")
async def create_session(request: Request) -> JSONResponse:
    if not app.state.http_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    try:
        body = await request.json()
    except Exception:
        body = {}
    lang = str(body.get("language", "pt")) if isinstance(body, dict) else "pt"
    if lang not in _ALLOWED_LANGS:
        lang = "pt"
    sid = app.state.memory.create_session(language=lang)
    return JSONResponse({"session_id": sid, "language": lang})


@app.get("/api/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> JSONResponse:
    if not app.state.http_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="invalid session id")
    data = app.state.memory.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="session not found")
    return JSONResponse(data)


@app.delete("/api/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> JSONResponse:
    if not app.state.http_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="invalid session id")
    app.state.memory.delete_session(session_id)
    return JSONResponse({"deleted": True})


@app.get("/api/sessions/{session_id}/export")
async def export_session(request: Request, session_id: str) -> PlainTextResponse:
    if not app.state.http_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="invalid session id")
    data = app.state.memory.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="session not found")
    parts = [s["text"] for s in data["segments"] if s.get("text")]
    text = " ".join(parts)
    filename = f"transcript-{session_id[:12]}.txt"
    return PlainTextResponse(
        text,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()

    client = ws.client.host if ws.client else "unknown"
    if not app.state.ws_limiter.allow(f"ws:{client}"):
        try:
            await ws.send_json({"type": "error", "message": "rate_limited"})
        except Exception:
            pass
        await ws.close(code=1013)
        return

    if not app.state.model_ready:
        try:
            await ws.send_json({"type": "error", "message": "model_not_ready"})
        except Exception:
            pass
        await ws.close(code=1011)
        return

    loop = app.state.loop
    session: StreamingSession | None = None

    def sender(msg: dict) -> None:
        try:
            asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, msg), loop)
        except Exception:
            logger.debug("ws send scheduling failed", exc_info=True)

    try:
        first = await ws.receive()
        if first.get("type") == "websocket.disconnect":
            return
        text_payload = first.get("text")
        if not text_payload:
            await ws.close(code=1003)
            return
        try:
            init = json.loads(text_payload)
        except Exception:
            await ws.send_json({"type": "error", "message": "invalid_init"})
            await ws.close(code=1003)
            return
        if not isinstance(init, dict) or init.get("type") != "start":
            await ws.send_json({"type": "error", "message": "expected_start"})
            await ws.close(code=1003)
            return

        lang = str(init.get("language", "pt"))
        if lang not in _ALLOWED_LANGS:
            lang = "pt"

        session_id = init.get("session_id")
        if session_id and (not validate_session_id(session_id) or not app.state.memory.session_exists(session_id)):
            session_id = None
        if not session_id:
            session_id = app.state.memory.create_session(language=lang)
        else:
            app.state.memory.set_session_language(session_id, lang)

        session = StreamingSession(
            session_id=session_id,
            language=lang,
            transcriber=app.state.transcriber,
            memory=app.state.memory,
            sender=sender,
        )

        await ws.send_json({
            "type": "ready",
            "session_id": session_id,
            "language": lang,
        })

        while True:
            msg = await ws.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                break
            chunk = msg.get("bytes")
            if chunk:
                if not app.state.msg_limiter.allow(f"ws:{client}"):
                    continue
                session.feed(chunk)
                continue
            text_msg = msg.get("text")
            if text_msg:
                try:
                    data = json.loads(text_msg)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                dtype = data.get("type")
                if dtype == "stop":
                    break
                if dtype == "language":
                    new_lang = str(data.get("language", lang))
                    if new_lang in _ALLOWED_LANGS:
                        lang = new_lang
                        session.language = new_lang
                        app.state.memory.set_session_language(session_id, new_lang)

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("ws stream error")
        try:
            await ws.send_json({"type": "error", "message": "internal_error"})
        except Exception:
            pass
    finally:
        if session is not None:
            try:
                await asyncio.to_thread(session.close)
                await asyncio.sleep(0.15)
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass


async def _safe_ws_send(ws: WebSocket, msg: dict) -> None:
    try:
        await ws.send_json(msg)
    except Exception:
        pass