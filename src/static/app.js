"use strict";

const STRINGS = {
  pt: {
    title: "Voice Notepad",
    ui_lang: "Interface",
    rec_lang: "Transcrição",
    record: "Gravar",
    stop: "Parar",
    copy: "Copiar",
    clear: "Limpar",
    export: "Exportar",
    history: "Histórico",
    ready: "Pronto",
    recording: "Gravando…",
    processing: "Processando…",
    loading: "Carregando modelo de IA…",
    words: "palavras",
    chars: "car",
    confidence: "Confiança",
    copied: "Copiado!",
    no_text: "Nada para copiar.",
    cleared: "Texto limpo.",
    placeholder: "Comece a gravar ou escreva aqui…",
    mic_error: "Microfone indisponível",
    model_error: "Modelo não carregado",
    rate_limited: "Limite de requisições atingido",
    filtered: "Conteúdo filtrado",
    err_insecure: "Microfone requer HTTPS ou http://localhost",
    err_no_api: "API de microfone indisponível neste navegador",
    err_server: "Falha ao conectar ao servidor",
    err_timeout: "Servidor não respondeu a tempo",
    err_denied: "Permissão negada. Verifique as permissões do microfone.",
    err_notfound: "Nenhum microfone encontrado",
    err_busy: "Microfone em uso por outro aplicativo",
    err_audio: "Áudio não suportado neste navegador",
    err_worklet: "Áudio avançado indisponível; usando modo de compatibilidade",
    err_ws: "Conexão com o servidor perdida",
  },
  en: {
    title: "Voice Notepad",
    ui_lang: "Interface",
    rec_lang: "Transcription",
    record: "Record",
    stop: "Stop",
    copy: "Copy",
    clear: "Clear",
    export: "Export",
    history: "History",
    ready: "Ready",
    recording: "Recording…",
    processing: "Processing…",
    loading: "Loading AI model…",
    words: "words",
    chars: "chars",
    confidence: "Confidence",
    copied: "Copied!",
    no_text: "Nothing to copy.",
    cleared: "Cleared.",
    placeholder: "Start recording or type here…",
    mic_error: "Microphone unavailable",
    model_error: "Model not ready",
    rate_limited: "Rate limit reached",
    filtered: "Content filtered",
    err_insecure: "Microphone requires HTTPS or http://localhost",
    err_no_api: "Microphone API unavailable in this browser",
    err_server: "Failed to connect to server",
    err_timeout: "Server did not respond in time",
    err_denied: "Permission denied. Check browser microphone permissions.",
    err_notfound: "No microphone found",
    err_busy: "Microphone is in use by another app",
    err_audio: "Audio not supported in this browser",
    err_worklet: "Advanced audio unavailable; using compatibility mode",
    err_ws: "Server connection lost",
  },
};

const state = {
  uiLang: "pt",
  recLang: "pt",
  theme: "light",
  recording: false,
  ws: null,
  sessionId: null,
  audioCtx: null,
  workletNode: null,
  mediaStream: null,
  sourceNode: null,
  finalizedUttIds: new Set(),
  pendingUttId: null,
  currentInterim: "",
};

let editor, committedEl, interimEl;

function t(key) {
  const lang = state.uiLang === "en" ? "en" : "pt";
  return (STRINGS[lang] && STRINGS[lang][key]) || key;
}

function el(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  el("status").textContent = text;
}

function toast(text) {
  const node = el("toast");
  node.textContent = text;
  node.classList.remove("hidden");
  requestAnimationFrame(() => node.classList.add("visible"));
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    node.classList.remove("visible");
    setTimeout(() => node.classList.add("hidden"), 220);
  }, 2600);
}

function applyUiLang() {
  document.documentElement.lang = state.uiLang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    node.textContent = t(key);
  });
  editor.setAttribute("data-placeholder", t("placeholder"));
  el("uiLang").value = state.uiLang;
  el("recLang").value = state.recLang;
  el("recordLabel").textContent = state.recording ? t("stop") : t("record");
}

function applyTheme() {
  document.body.setAttribute("data-theme", state.theme);
}

function getCommittedText() {
  return committedEl.textContent || "";
}

function getInterimText() {
  return interimEl.textContent || "";
}

function getDisplayText() {
  const c = getCommittedText();
  const i = getInterimText();
  if (!i) return c;
  if (!c) return i;
  return /\s$/.test(c) ? c + i : c + " " + i;
}

function updateCounts() {
  const text = getDisplayText();
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  el("counts").textContent = `${words} ${t("words")}  ·  ${text.length} ${t("chars")}`;
}

function updateEditorPlaceholder() {
  const isEmpty = !getCommittedText() && !getInterimText();
  editor.classList.toggle("empty", isEmpty);
}

function nearBottom(threshold) {
  return editor.scrollHeight - editor.scrollTop - editor.clientHeight < threshold;
}

function autoScroll() {
  editor.scrollTop = editor.scrollHeight;
}

function setInterim(text) {
  state.currentInterim = text || "";
  if (interimEl.textContent !== state.currentInterim) {
    interimEl.textContent = state.currentInterim;
  }
  updateEditorPlaceholder();
  updateCounts();
  autoScroll();
}

function clearInterim() {
  state.currentInterim = "";
  if (interimEl.textContent) {
    interimEl.textContent = "";
  }
  updateEditorPlaceholder();
  updateCounts();
}

function appendCommittedText(text) {
  if (!text) return;
  const cur = getCommittedText();
  const sep = cur && !/\s$/.test(cur) ? " " : "";
  committedEl.appendChild(document.createTextNode(sep + text));
  updateEditorPlaceholder();
  updateCounts();
  autoScroll();
}

function setCommittedText(text) {
  committedEl.textContent = text || "";
  updateEditorPlaceholder();
  updateCounts();
}

function ensureEditorStructure() {
  if (interimEl.parentNode !== editor) {
    editor.appendChild(interimEl);
  }
  if (committedEl.parentNode !== editor) {
    editor.insertBefore(committedEl, editor.firstChild);
  }
  if (interimEl.nextSibling) {
    while (interimEl.nextSibling) editor.removeChild(interimEl.nextSibling);
  }
}

function setConfidence(value) {
  const pct = Math.max(0, Math.min(1, value || 0));
  const fill = el("confFill");
  fill.style.width = `${Math.round(pct * 100)}%`;
  let colour = "var(--red)";
  if (pct >= 0.7) colour = "var(--green)";
  else if (pct >= 0.4) colour = "var(--amber)";
  fill.style.background = colour;
  el("confValue").textContent = `${Math.round(pct * 100)}%`;
}

function setVadDot(name) {
  const dot = el("vadDot");
  dot.classList.remove("listening", "speaking");
  if (name === "listening") dot.classList.add("listening");
  else if (name === "speaking") dot.classList.add("speaking");
}

function downsample(buffer, inRate, outRate) {
  if (inRate === outRate) return buffer;
  const ratio = inRate / outRate;
  const outLen = Math.floor(buffer.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, buffer.length - 1);
    const frac = pos - i0;
    out[i] = buffer[i0] * (1 - frac) + buffer[i1] * frac;
  }
  return out;
}

function floatToInt16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    let s = f32[i];
    if (s > 1) s = 1; else if (s < -1) s = -1;
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/stream`;
}

function withTimeout(promise, ms) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("timeout")), ms);
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); }
    );
  });
}

function connectWebSocket() {
  return new Promise((resolve, reject) => {
    let ws;
    try {
      ws = new WebSocket(wsUrl());
    } catch (e) {
      reject(new Error("ws_construct"));
      return;
    }
    ws.binaryType = "arraybuffer";

    let settled = false;
    const settleOk = () => {
      if (settled) return;
      settled = true;
      resolve(ws);
    };
    const settleErr = (reason) => {
      if (settled) return;
      settled = true;
      reject(new Error(reason));
    };

    ws.onopen = () => {
      try {
        ws.send(JSON.stringify({ type: "start", language: state.recLang }));
      } catch (e) {
        settleErr("ws_send_failed");
      }
    };

    ws.onerror = () => settleErr("ws_error");

    ws.onclose = () => {
      if (state.ws === ws) state.ws = null;
      if (!settled) settleErr("ws_closed");
      else if (state.recording) {
        toast(t("err_ws"));
        stopRecording();
      }
    };

    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      if (msg.type === "ready") {
        state.sessionId = msg.session_id;
        settleOk();
      } else if (msg.type === "error" && !settled) {
        settleErr(msg.message || "server_error");
      }
      handleServerMessage(msg);
    };

    state.ws = ws;
  });
}

function handleServerMessage(msg) {
  switch (msg.type) {
    case "ready":
      setStatus(t("recording"));
      break;
    case "partial":
      if (state.finalizedUttIds.has(msg.utt_id)) return;
      state.pendingUttId = msg.utt_id;
      setInterim(msg.text);
      if (typeof msg.confidence === "number") setConfidence(msg.confidence);
      break;
    case "final":
      if (state.finalizedUttIds.has(msg.utt_id)) return;
      state.finalizedUttIds.add(msg.utt_id);
      if (state.pendingUttId === msg.utt_id) state.pendingUttId = null;
      clearInterim();
      appendCommittedText(msg.text);
      if (typeof msg.confidence === "number") setConfidence(msg.confidence);
      break;
    case "vad":
      if (state.recording) setVadDot(msg.active ? "speaking" : "listening");
      break;
    case "warning":
      if (msg.message === "content_filtered") toast(t("filtered"));
      break;
    case "error":
      if (msg.message === "rate_limited") toast(t("rate_limited"));
      else if (msg.message === "model_not_ready") toast(t("model_error"));
      else toast(msg.message || "error");
      break;
  }
}

function handleMicError(err) {
  const name = (err && err.name) || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
    toast(t("err_denied"));
  } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    toast(t("err_notfound"));
  } else if (name === "NotReadableError" || name === "TrackStartError") {
    toast(t("err_busy"));
  } else if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    toast(t("err_denied"));
  } else if (name === "AbortError") {
    toast(t("err_busy"));
  } else {
    toast(t("mic_error"));
  }
}

async function createAudioPipeline(stream) {
  let ctx;
  try {
    ctx = new AudioContext({ sampleRate: 16000 });
  } catch (e) {
    try {
      ctx = new AudioContext();
    } catch (e2) {
      return { error: "audio_unsupported" };
    }
  }
  if (ctx.state === "suspended") {
    try { await ctx.resume(); } catch (e) {}
  }
  state.audioCtx = ctx;

  const nativeRate = ctx.sampleRate;
  const source = ctx.createMediaStreamSource(stream);
  state.sourceNode = source;

  const sink = ctx.createGain();
  sink.gain.value = 0;

  let workletOk = true;
  let node = null;
  try {
    if (!ctx.audioWorklet) throw new Error("no_worklet");
    await ctx.audioWorklet.addModule("/static/capture-processor.js");
    node = new AudioWorkletNode(ctx, "capture-processor");
  } catch (e) {
    workletOk = false;
  }

  if (workletOk && node) {
    node.port.onmessage = (event) => {
      if (!state.recording) return;
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
      const chunk = event.data;
      if (!chunk || !chunk.length) return;
      const down = downsample(chunk, nativeRate, 16000);
      const i16 = floatToInt16(down);
      try { state.ws.send(i16.buffer); } catch (e) {}
    };
    source.connect(node);
    node.connect(sink).connect(ctx.destination);
    state.workletNode = node;
    return { error: null, mode: "worklet" };
  }

  try {
    const sp = ctx.createScriptProcessor(1024, 1, 1);
    sp.onaudioprocess = (event) => {
      if (!state.recording) return;
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const down = downsample(input, nativeRate, 16000);
      const i16 = floatToInt16(down);
      try { state.ws.send(i16.buffer); } catch (e) {}
    };
    source.connect(sp);
    sp.connect(sink).connect(ctx.destination);
    state.workletNode = sp;
    return { error: null, mode: "script" };
  } catch (e) {
    return { error: "audio_unsupported" };
  }
}

async function startRecording() {
  if (state.recording) return;

  if (!window.isSecureContext) {
    toast(t("err_insecure"));
    return;
  }
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    toast(t("err_no_api"));
    return;
  }

  if (state.ws) {
    try { state.ws.close(); } catch (e) {}
    state.ws = null;
  }
  state.finalizedUttIds = new Set();
  state.pendingUttId = null;
  clearInterim();

  let ws;
  try {
    ws = await withTimeout(connectWebSocket(), 8000);
  } catch (err) {
    const reason = err && err.message;
    if (reason === "timeout") toast(t("err_timeout"));
    else if (reason === "model_not_ready") toast(t("model_error"));
    else if (reason === "rate_limited") toast(t("rate_limited"));
    else toast(t("err_server"));
    try { if (state.ws) state.ws.close(); } catch (e) {}
    state.ws = null;
    return;
  }
  state.ws = ws;

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
  } catch (err) {
    handleMicError(err);
    try { state.ws.close(); } catch (e) {}
    state.ws = null;
    return;
  }
  state.mediaStream = stream;

  const pipeline = await createAudioPipeline(stream);
  if (pipeline.error) {
    toast(t("err_audio"));
    await teardownAudio();
    try { state.ws.close(); } catch (e) {}
    state.ws = null;
    return;
  }
  if (pipeline.mode === "script") {
    toast(t("err_worklet"));
  }

  state.recording = true;
  el("recordBtn").classList.add("recording");
  el("recordLabel").textContent = t("stop");
  setStatus(t("recording"));
  setVadDot("listening");
}

async function teardownAudio() {
  try { if (state.workletNode) state.workletNode.disconnect(); } catch (e) {}
  try { if (state.sourceNode) state.sourceNode.disconnect(); } catch (e) {}
  try {
    if (state.mediaStream) {
      state.mediaStream.getTracks().forEach((tr) => tr.stop());
    }
  } catch (e) {}
  try {
    if (state.audioCtx && state.audioCtx.state !== "closed") {
      await state.audioCtx.close();
    }
  } catch (e) {}
  state.workletNode = null;
  state.sourceNode = null;
  state.mediaStream = null;
  state.audioCtx = null;
}

async function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  el("recordBtn").classList.remove("recording");
  el("recordLabel").textContent = t("record");
  setVadDot(null);
  setStatus(t("processing"));

  const ws = state.ws;
  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  } catch (e) {}

  await teardownAudio();

  setTimeout(() => {
    if (!state.recording) {
      setStatus(t("ready"));
      setVadDot(null);
      clearInterim();
    }
  }, 1200);

  if (ws) {
    setTimeout(() => {
      try { if (ws.readyState === WebSocket.OPEN) ws.close(); } catch (e) {}
      if (state.ws === ws) state.ws = null;
    }, 1800);
  }
}

async function copyToClipboard() {
  const text = getDisplayText();
  if (!text.trim()) {
    toast(t("no_text"));
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    toast(t("copied"));
  } catch (e) {
    const range = document.createRange();
    range.selectNodeContents(editor);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    try { document.execCommand("copy"); toast(t("copied")); } catch (e2) { toast("error"); }
    sel.removeAllRanges();
  }
}

function clearAll() {
  setCommittedText("");
  clearInterim();
  setConfidence(0);
  editor.classList.add("empty");
  toast(t("cleared"));
}

function exportSession() {
  if (!state.sessionId) {
    const text = getDisplayText();
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
    return;
  }
  window.location.href = `/api/sessions/${encodeURIComponent(state.sessionId)}/export`;
}

async function loadHistory() {
  try {
    const r = await fetch("/api/sessions?limit=50");
    const data = await r.json();
    const list = el("historyList");
    list.innerHTML = "";
    (data.sessions || []).forEach((s) => {
      const li = document.createElement("li");
      const date = new Date(s.updated_at * 1000);
      li.innerHTML = `<div>${s.id.slice(0, 12)}…</div><div class="meta">${date.toLocaleString()} · ${s.language}</div>`;
      li.addEventListener("click", () => loadSession(s.id));
      list.appendChild(li);
    });
  } catch (e) {
    toast("error");
  }
  el("historyPanel").classList.remove("hidden");
}

async function loadSession(sessionId) {
  try {
    const r = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    if (!r.ok) return;
    const data = await r.json();
    const text = (data.segments || [])
      .filter((s) => s.text)
      .map((s) => s.text)
      .join(" ");
    setCommittedText(text);
    clearInterim();
    state.sessionId = sessionId;
    state.finalizedUttIds = new Set();
    el("historyPanel").classList.add("hidden");
    autoScroll();
  } catch (e) {
    toast("error");
  }
}

function bindEditorEvents() {
  editor.addEventListener("input", () => {
    ensureEditorStructure();
    if (interimEl.textContent !== state.currentInterim) {
      interimEl.textContent = state.currentInterim;
    }
    updateEditorPlaceholder();
    updateCounts();
  });

  editor.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey) {
      e.preventDefault();
      try {
        document.execCommand("insertText", false, "\n");
      } catch (err) {
        const sel = window.getSelection();
        if (sel && sel.rangeCount) {
          const range = sel.getRangeAt(0);
          range.deleteContents();
          const node = document.createTextNode("\n");
          range.insertNode(node);
          range.setStartAfter(node);
          range.collapse(true);
          sel.removeAllRanges();
          sel.addRange(range);
        }
      }
      updateCounts();
      updateEditorPlaceholder();
      autoScroll();
    }
  });

  editor.addEventListener("paste", (e) => {
    e.preventDefault();
    const clipboard = e.clipboardData || window.clipboardData;
    if (!clipboard) return;
    let text = clipboard.getData("text/plain") || "";
    if (text.length > 500000) text = text.slice(0, 500000);
    try {
      document.execCommand("insertText", false, text);
    } catch (err) {
      const sel = window.getSelection();
      if (sel && sel.rangeCount) {
        const range = sel.getRangeAt(0);
        range.deleteContents();
        const node = document.createTextNode(text);
        range.insertNode(node);
        range.setStartAfter(node);
        range.collapse(true);
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }
    ensureEditorStructure();
    updateEditorPlaceholder();
    updateCounts();
  });

  editor.addEventListener("scroll", () => {});
}

function bindEvents() {
  el("recordBtn").addEventListener("click", async () => {
    if (state.recording) await stopRecording();
    else await startRecording();
  });

  el("copyBtn").addEventListener("click", copyToClipboard);
  el("clearBtn").addEventListener("click", clearAll);
  el("exportBtn").addEventListener("click", exportSession);

  el("themeBtn").addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
  });

  el("uiLang").addEventListener("change", (e) => {
    state.uiLang = e.target.value === "en" ? "en" : "pt";
    applyUiLang();
  });

  el("recLang").addEventListener("change", (e) => {
    const v = e.target.value;
    if (["pt", "en", "auto"].includes(v)) {
      state.recLang = v;
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        try { state.ws.send(JSON.stringify({ type: "language", language: v })); } catch (err) {}
      }
    }
  });

  el("historyBtn").addEventListener("click", loadHistory);
  el("historyClose").addEventListener("click", () => {
    el("historyPanel").classList.add("hidden");
  });
}

function init() {
  editor = el("editor");
  committedEl = el("committed");
  interimEl = el("interim");

  applyTheme();
  applyUiLang();
  bindEditorEvents();
  bindEvents();
  updateEditorPlaceholder();
  updateCounts();
  setConfidence(0);

  if (!window.isSecureContext) {
    setStatus(t("err_insecure"));
    return;
  }
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    setStatus(t("err_no_api"));
    return;
  }

  fetch("/api/health")
    .then((r) => r.json())
    .then((d) => setStatus(d.model_ready ? t("ready") : t("loading")))
    .catch(() => setStatus(t("ready")));
}

document.addEventListener("DOMContentLoaded", init);