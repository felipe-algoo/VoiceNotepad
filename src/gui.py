import logging
import pathlib
import threading
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
import wx
import wx.adv

from src.audio_engine import AudioEngine
from src.file_manager import FileManager
from src.transcriber import Transcriber, TranscriptionWorker
from src.security import sanitize_transcription, check_prompt_injection, safe_error_message, content_fingerprint

logger = logging.getLogger(__name__)

_MODEL_SIZE: Final[str] = "base"
_PULSE_MS: Final[int] = 550
_GAUGE_MS: Final[int] = 55
_MIN_W: Final[int] = 600
_MIN_H: Final[int] = 700
_MAX_EDITOR_CHARS: Final[int] = 500_000
_FLASH_MS: Final[int] = 4000
_UNDO_LIMIT: Final[int] = 200

_ID_TOGGLE_THEME: int = wx.NewIdRef(count=1)
_ID_REC_PT: int = wx.NewIdRef(count=1)
_ID_REC_EN: int = wx.NewIdRef(count=1)
_ID_REC_AUTO: int = wx.NewIdRef(count=1)
_ID_UI_PT: int = wx.NewIdRef(count=1)
_ID_UI_EN: int = wx.NewIdRef(count=1)
_ID_WORD_WRAP: int = wx.NewIdRef(count=1)
_ID_FONT_INC: int = wx.NewIdRef(count=1)
_ID_FONT_DEC: int = wx.NewIdRef(count=1)

_STRINGS: dict[str, dict[str, str]] = {
    "pt": {
        "title": "Bloco de Notas por Voz",
        "menu_file": "&Arquivo",
        "menu_new": "&Novo\tCtrl+N",
        "menu_open": "&Abrir...\tCtrl+O",
        "menu_save": "&Salvar\tCtrl+S",
        "menu_save_as": "Salvar &Como...",
        "menu_exit": "Sai&r\tAlt+F4",
        "menu_view": "&Exibir",
        "menu_theme": "Alternar &Tema\tCtrl+T",
        "menu_ui_lang": "Idioma da &Interface",
        "menu_rec_lang": "Idioma de &Transcrição",
        "menu_lang_pt": "Português (BR)",
        "menu_lang_en": "English",
        "menu_rec_auto": "Automático",
        "menu_word_wrap": "Quebra de &Linha",
        "menu_font_inc": "Aumentar Fonte\tCtrl+=",
        "menu_font_dec": "Diminuir Fonte\tCtrl+-",
        "lbl_trans_lang": "Transcrição:",
        "lbl_ui_lang": "Interface:",
        "btn_save": "Salvar",
        "btn_open": "Abrir",
        "btn_clear": "Limpar",
        "btn_copy": "Copiar",
        "btn_record": "  ● Gravar  ",
        "btn_stop": "  ■ Parar  ",
        "hint": "Comece a gravar ou escreva aqui...",
        "ready": "Pronto",
        "recording": "Gravando…",
        "processing": "Processando…",
        "loading": "Carregando modelo de IA…",
        "loading_sub": "Isso pode levar alguns segundos na primeira execução.",
        "loading_dl": "Baixando modelo base (~150 MB) na primeira vez…",
        "words": "palavras",
        "chars": "car",
        "dlg_clear_title": "Limpar Tudo?",
        "dlg_clear_body": "Todo o texto será apagado permanentemente.",
        "msg_copied": "Copiado para a área de transferência!",
        "msg_saved": "Arquivo salvo com sucesso.",
        "msg_no_text": "Nenhum texto para copiar.",
        "msg_mic_error": "Microfone indisponível",
        "msg_model_error": "Falha ao carregar o modelo de IA.",
        "msg_error": "Erro",
        "msg_injection_warning": "Conteúdo suspeito detectado e filtrado.",
        "msg_too_large": "Conteúdo muito grande para adicionar.",
        "lbl_untitled": "Sem título",
        "shortcuts": "Ctrl+R: Gravar  ·  Ctrl+S: Salvar  ·  Ctrl+O: Abrir  ·  Ctrl+T: Tema",
        "dlg_save_title": "Salvar Nota",
        "dlg_open_title": "Abrir Nota",
        "file_filter": "Arquivos de texto (*.txt;*.md)|*.txt;*.md|Todos os arquivos (*.*)|*.*",
        "status_ready": "Pronto",
        "btn_theme_tip": "Alternar tema claro/escuro",
        "tray_tooltip": "Bloco de Notas por Voz",
        "tray_show": "Mostrar",
        "tray_exit": "Sair",
        "confidence": "Confiança",
        "lbl_confidence": "Conf.:",
        "warn_unsaved": "Há alterações não salvas. Deseja continuar?",
        "warn_unsaved_title": "Alterações não salvas",
    },
    "en": {
        "title": "Voice Notepad",
        "menu_file": "&File",
        "menu_new": "&New\tCtrl+N",
        "menu_open": "&Open...\tCtrl+O",
        "menu_save": "&Save\tCtrl+S",
        "menu_save_as": "Save &As...",
        "menu_exit": "E&xit\tAlt+F4",
        "menu_view": "&View",
        "menu_theme": "Toggle &Theme\tCtrl+T",
        "menu_ui_lang": "&Interface Language",
        "menu_rec_lang": "&Transcription Language",
        "menu_lang_pt": "Português (BR)",
        "menu_lang_en": "English",
        "menu_rec_auto": "Auto",
        "menu_word_wrap": "&Word Wrap",
        "menu_font_inc": "Increase Font\tCtrl+=",
        "menu_font_dec": "Decrease Font\tCtrl+-",
        "lbl_trans_lang": "Transcription:",
        "lbl_ui_lang": "Interface:",
        "btn_save": "Save",
        "btn_open": "Open",
        "btn_clear": "Clear",
        "btn_copy": "Copy",
        "btn_record": "  ● Record  ",
        "btn_stop": "  ■ Stop  ",
        "hint": "Start recording or type here…",
        "ready": "Ready",
        "recording": "Recording…",
        "processing": "Processing…",
        "loading": "Loading AI model…",
        "loading_sub": "This may take a few seconds on first run.",
        "loading_dl": "Downloading base model (~150 MB) on first run…",
        "words": "words",
        "chars": "chars",
        "dlg_clear_title": "Clear Everything?",
        "dlg_clear_body": "All text will be permanently erased.",
        "msg_copied": "Copied to clipboard!",
        "msg_saved": "File saved successfully.",
        "msg_no_text": "No text to copy.",
        "msg_mic_error": "Microphone unavailable",
        "msg_model_error": "Failed to load the AI model.",
        "msg_error": "Error",
        "msg_injection_warning": "Suspicious content detected and filtered.",
        "msg_too_large": "Content too large to append.",
        "lbl_untitled": "Untitled",
        "shortcuts": "Ctrl+R: Record  ·  Ctrl+S: Save  ·  Ctrl+O: Open  ·  Ctrl+T: Theme",
        "dlg_save_title": "Save Note",
        "dlg_open_title": "Open Note",
        "file_filter": "Text files (*.txt;*.md)|*.txt;*.md|All files (*.*)|*.*",
        "status_ready": "Ready",
        "btn_theme_tip": "Toggle light/dark theme",
        "tray_tooltip": "Voice Notepad",
        "tray_show": "Show",
        "tray_exit": "Exit",
        "confidence": "Confidence",
        "lbl_confidence": "Conf.:",
        "warn_unsaved": "There are unsaved changes. Continue anyway?",
        "warn_unsaved_title": "Unsaved Changes",
    },
}

_REC_LANG_CODES: Final[list[str]] = ["pt", "en", "auto"]
_UI_LANG_CODES: Final[list[str]] = ["pt", "en"]
_FONT_SIZE_MIN: Final[int] = 8
_FONT_SIZE_MAX: Final[int] = 28
_FONT_SIZE_DEFAULT: Final[int] = 12


@dataclass(frozen=True)
class _Theme:
    is_dark: bool
    bg: wx.Colour
    fg: wx.Colour
    bar_bg: wx.Colour
    bar_fg: wx.Colour
    primary: wx.Colour
    primary_fg: wx.Colour
    stop_bg: wx.Colour
    stop_fg: wx.Colour
    input_bg: wx.Colour
    input_fg: wx.Colour
    sec_btn_bg: wx.Colour
    sec_btn_fg: wx.Colour
    outline: wx.Colour
    vad_idle: wx.Colour
    vad_listening: wx.Colour
    vad_speaking: wx.Colour
    hint_fg: wx.Colour
    conf_low: wx.Colour
    conf_mid: wx.Colour
    conf_high: wx.Colour


_LIGHT = _Theme(
    is_dark=False,
    bg=wx.Colour(255, 251, 254),
    fg=wx.Colour(28, 27, 31),
    bar_bg=wx.Colour(235, 228, 242),
    bar_fg=wx.Colour(73, 69, 79),
    primary=wx.Colour(103, 80, 164),
    primary_fg=wx.Colour(255, 255, 255),
    stop_bg=wx.Colour(179, 38, 30),
    stop_fg=wx.Colour(255, 255, 255),
    input_bg=wx.Colour(255, 255, 255),
    input_fg=wx.Colour(28, 27, 31),
    sec_btn_bg=wx.Colour(231, 224, 236),
    sec_btn_fg=wx.Colour(103, 80, 164),
    outline=wx.Colour(202, 196, 208),
    vad_idle=wx.Colour(176, 170, 182),
    vad_listening=wx.Colour(245, 158, 11),
    vad_speaking=wx.Colour(34, 197, 94),
    hint_fg=wx.Colour(160, 156, 166),
    conf_low=wx.Colour(179, 38, 30),
    conf_mid=wx.Colour(245, 158, 11),
    conf_high=wx.Colour(34, 197, 94),
)

_DARK = _Theme(
    is_dark=True,
    bg=wx.Colour(28, 27, 31),
    fg=wx.Colour(230, 225, 229),
    bar_bg=wx.Colour(43, 41, 48),
    bar_fg=wx.Colour(202, 196, 208),
    primary=wx.Colour(208, 188, 255),
    primary_fg=wx.Colour(56, 30, 114),
    stop_bg=wx.Colour(242, 184, 181),
    stop_fg=wx.Colour(96, 20, 16),
    input_bg=wx.Colour(49, 48, 51),
    input_fg=wx.Colour(230, 225, 229),
    sec_btn_bg=wx.Colour(73, 69, 79),
    sec_btn_fg=wx.Colour(208, 188, 255),
    outline=wx.Colour(80, 76, 86),
    vad_idle=wx.Colour(100, 96, 108),
    vad_listening=wx.Colour(245, 158, 11),
    vad_speaking=wx.Colour(34, 197, 94),
    hint_fg=wx.Colour(110, 106, 118),
    conf_low=wx.Colour(242, 184, 181),
    conf_mid=wx.Colour(254, 215, 170),
    conf_high=wx.Colour(134, 239, 172),
)


def _ui_font(size: int = 10, bold: bool = False) -> wx.Font:
    f = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
    f.SetPointSize(size)
    f.SetWeight(wx.FONTWEIGHT_BOLD if bold else wx.FONTWEIGHT_NORMAL)
    return f


class _VadDot(wx.Window):
    _D: Final[int] = 12

    def __init__(self, parent: wx.Window) -> None:
        super().__init__(parent, size=(self._D, self._D))
        self.SetMinSize((self._D, self._D))
        self._colour = _LIGHT.vad_idle
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda _: None)

    def set_colour(self, colour: wx.Colour) -> None:
        self._colour = colour
        self.Refresh()

    def _on_paint(self, _: wx.PaintEvent) -> None:
        dc = wx.PaintDC(self)
        parent_bg = self.GetParent().GetBackgroundColour()
        dc.SetBackground(wx.Brush(parent_bg))
        dc.Clear()
        dc.SetBrush(wx.Brush(self._colour))
        dc.SetPen(wx.TRANSPARENT_PEN)
        r = self._D // 2
        dc.DrawCircle(r, r, r)


class _ConfidenceBadge(wx.Window):
    _W: Final[int] = 52
    _H: Final[int] = 18

    def __init__(self, parent: wx.Window, theme: "_Theme") -> None:
        super().__init__(parent, size=(self._W, self._H))
        self.SetMinSize((self._W, self._H))
        self._theme = theme
        self._value: float = 0.0
        self._visible: bool = False
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda _: None)

    def update(self, conf: float, theme: "_Theme") -> None:
        self._theme = theme
        self._value = max(0.0, min(1.0, conf))
        self._visible = True
        self.Refresh()

    def hide_badge(self) -> None:
        self._visible = False
        self.Refresh()

    def _on_paint(self, _: wx.PaintEvent) -> None:
        dc = wx.PaintDC(self)
        parent_bg = self.GetParent().GetBackgroundColour()
        dc.SetBackground(wx.Brush(parent_bg))
        dc.Clear()
        if not self._visible:
            return
        t = self._theme
        if self._value < 0.4:
            colour = t.conf_low
        elif self._value < 0.7:
            colour = t.conf_mid
        else:
            colour = t.conf_high
        dc.SetBrush(wx.Brush(colour))
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.DrawRoundedRectangle(0, 0, self._W, self._H, 4)
        dc.SetTextForeground(wx.Colour(255, 255, 255))
        dc.SetFont(_ui_font(8, bold=True))
        label = f"{int(self._value * 100)}%"
        tw, th = dc.GetTextExtent(label)
        dc.DrawText(label, (self._W - tw) // 2, (self._H - th) // 2)


class _LoadingPanel(wx.Panel):
    def __init__(self, parent: wx.Window, theme: "_Theme", lang: str) -> None:
        super().__init__(parent)
        self._theme = theme
        self._lang = lang
        self._gauge: Optional[wx.Gauge] = None
        self._lbl_main: Optional[wx.StaticText] = None
        self._lbl_sub: Optional[wx.StaticText] = None
        self._lbl_dl: Optional[wx.StaticText] = None
        self._gauge_timer: Optional[wx.Timer] = None
        self._build()
        self._apply_theme(theme)

    def _t(self, key: str) -> str:
        return _STRINGS.get(self._lang, _STRINGS["en"]).get(key, key)

    def _build(self) -> None:
        outer = wx.BoxSizer(wx.VERTICAL)
        inner = wx.BoxSizer(wx.VERTICAL)

        self._gauge = wx.Gauge(self, range=100, size=(260, 6), style=wx.GA_HORIZONTAL)

        self._lbl_main = wx.StaticText(self, label=self._t("loading"), style=wx.ALIGN_CENTER)
        self._lbl_main.SetFont(_ui_font(14, bold=True))

        self._lbl_sub = wx.StaticText(self, label=self._t("loading_sub"), style=wx.ALIGN_CENTER)
        self._lbl_sub.SetFont(_ui_font(10))

        self._lbl_dl = wx.StaticText(self, label=self._t("loading_dl"), style=wx.ALIGN_CENTER)
        self._lbl_dl.SetFont(_ui_font(9))

        inner.Add(self._gauge, 0, wx.ALIGN_CENTER | wx.BOTTOM, 28)
        inner.Add(self._lbl_main, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        inner.Add(self._lbl_sub, 0, wx.ALIGN_CENTER | wx.BOTTOM, 6)
        inner.Add(self._lbl_dl, 0, wx.ALIGN_CENTER)

        outer.AddStretchSpacer()
        outer.Add(inner, 0, wx.ALIGN_CENTER | wx.ALL, 40)
        outer.AddStretchSpacer()
        self.SetSizer(outer)

        self._gauge_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_gauge_tick, self._gauge_timer)
        self._gauge_timer.Start(_GAUGE_MS)

    def _on_gauge_tick(self, _: wx.TimerEvent) -> None:
        if self._gauge:
            self._gauge.Pulse()

    def stop_animation(self) -> None:
        if self._gauge_timer and self._gauge_timer.IsRunning():
            self._gauge_timer.Stop()

    def show_error(self, exc: Exception) -> None:
        self.stop_animation()
        safe_msg = safe_error_message(exc)
        if self._gauge:
            self._gauge.Hide()
        if self._lbl_main:
            self._lbl_main.SetForegroundColour(wx.Colour(179, 38, 30))
            self._lbl_main.SetLabel(_STRINGS.get(self._lang, _STRINGS["en"])["msg_model_error"])
        if self._lbl_sub:
            self._lbl_sub.SetLabel(safe_msg)
        if self._lbl_dl:
            self._lbl_dl.SetLabel("Run: pip install faster-whisper")
        self.Layout()

    def _apply_theme(self, theme: "_Theme") -> None:
        self._theme = theme
        self.SetBackgroundColour(theme.bg)
        for w in (self._lbl_main, self._lbl_sub, self._lbl_dl):
            if w:
                w.SetBackgroundColour(theme.bg)
                w.SetForegroundColour(theme.fg)
        if self._lbl_sub:
            self._lbl_sub.SetForegroundColour(theme.bar_fg)
        if self._lbl_dl:
            self._lbl_dl.SetForegroundColour(theme.hint_fg)
        self.Refresh()


class MainFrame(wx.Frame):
    def __init__(self) -> None:
        super().__init__(
            None,
            title=_STRINGS["pt"]["title"],
            size=(_MIN_W, _MIN_H),
        )
        self.SetMinSize((_MIN_W, _MIN_H))

        self._ui_lang: str = "pt"
        self._rec_lang: str = "pt"
        self._recording: bool = False
        self._model_ready: bool = False
        self._unsaved: bool = False
        self._theme: _Theme = _LIGHT
        self._pulse_on: bool = True
        self._font_size: int = _FONT_SIZE_DEFAULT
        self._word_wrap: bool = True
        self._last_conf: float = 0.0
        self._injection_warnings: int = 0

        self._transcriber = Transcriber(language=self._rec_lang)
        self._file_manager = FileManager()
        self._engine: Optional[AudioEngine] = None
        self._worker: Optional[TranscriptionWorker] = None

        self._loading_panel: Optional[_LoadingPanel] = None
        self._editor_panel: Optional[wx.Panel] = None
        self._editor: Optional[wx.TextCtrl] = None
        self._rec_btn: Optional[wx.Button] = None
        self._vad_dot: Optional[_VadDot] = None
        self._conf_badge: Optional[_ConfidenceBadge] = None
        self._status_bar: Optional[wx.StatusBar] = None
        self._pulse_timer: Optional[wx.Timer] = None
        self._rec_lang_choice: Optional[wx.Choice] = None
        self._ui_lang_choice: Optional[wx.Choice] = None
        self._mi_rec_pt: Optional[wx.MenuItem] = None
        self._mi_rec_en: Optional[wx.MenuItem] = None
        self._mi_rec_auto: Optional[wx.MenuItem] = None
        self._mi_ui_pt: Optional[wx.MenuItem] = None
        self._mi_ui_en: Optional[wx.MenuItem] = None
        self._mi_word_wrap: Optional[wx.MenuItem] = None
        self._hint_active: bool = True
        self._btn_save: Optional[wx.Button] = None
        self._btn_open: Optional[wx.Button] = None
        self._btn_clear: Optional[wx.Button] = None
        self._btn_copy: Optional[wx.Button] = None

        self._build_ui()
        self._bind_events()
        self._start_model_load()

    def _t(self, key: str) -> str:
        return _STRINGS.get(self._ui_lang, _STRINGS["en"]).get(key, key)

    def _build_ui(self) -> None:
        self._status_bar = self.CreateStatusBar(3)
        self._status_bar.SetStatusWidths([-2, -4, -2])
        self._status_bar.SetStatusText(self._t("status_ready"), 0)
        self._status_bar.SetStatusText(self._t("lbl_untitled"), 1)
        self._status_bar.SetStatusText("0 " + self._t("words"), 2)
        self._status_bar.SetBackgroundColour(self._theme.bar_bg)
        self._status_bar.SetForegroundColour(self._theme.bar_fg)

        root = wx.Panel(self)
        root.SetBackgroundColour(self._theme.bg)
        root_sizer = wx.BoxSizer(wx.VERTICAL)

        self._loading_panel = _LoadingPanel(root, self._theme, self._ui_lang)
        self._editor_panel = self._build_editor(root)
        self._editor_panel.Hide()

        root_sizer.Add(self._loading_panel, 1, wx.EXPAND)
        root_sizer.Add(self._editor_panel, 1, wx.EXPAND)
        root.SetSizer(root_sizer)

        self._build_menu()

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root, 1, wx.EXPAND)
        self.SetSizer(frame_sizer)
        self.Layout()

    def _build_editor(self, parent: wx.Window) -> wx.Panel:
        panel = wx.Panel(parent)
        panel.SetBackgroundColour(self._theme.bg)
        sizer = wx.BoxSizer(wx.VERTICAL)

        lang_bar = self._build_lang_bar(panel)
        sizer.Add(lang_bar, 0, wx.EXPAND)

        sizer.Add(
            wx.StaticLine(panel, style=wx.LI_HORIZONTAL),
            0, wx.EXPAND,
        )

        editor_style = wx.TE_MULTILINE | wx.BORDER_NONE
        if self._word_wrap:
            editor_style |= wx.TE_WORDWRAP
        else:
            editor_style |= wx.TE_DONTWRAP

        self._editor = wx.TextCtrl(
            panel,
            style=editor_style,
        )
        self._editor.SetFont(_ui_font(self._font_size))
        self._editor.SetBackgroundColour(self._theme.input_bg)
        self._editor.SetForegroundColour(self._theme.hint_fg)
        self._editor.SetHint(self._t("hint"))
        self._editor.SetMaxLength(0)
        sizer.Add(self._editor, 1, wx.EXPAND | wx.ALL, 12)

        sizer.Add(
            wx.StaticLine(panel, style=wx.LI_HORIZONTAL),
            0, wx.EXPAND,
        )

        btn_bar = self._build_btn_bar(panel)
        sizer.Add(btn_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)

        rec_row = self._build_rec_row(panel)
        sizer.Add(rec_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)

        shortcuts = wx.StaticText(panel, label=self._t("shortcuts"), style=wx.ALIGN_CENTER)
        shortcuts.SetFont(_ui_font(9))
        shortcuts.SetForegroundColour(self._theme.hint_fg)
        shortcuts.SetBackgroundColour(self._theme.bg)
        sizer.Add(shortcuts, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 6)

        panel.SetSizer(sizer)
        return panel

    def _build_lang_bar(self, parent: wx.Window) -> wx.Panel:
        bar = wx.Panel(parent)
        bar.SetBackgroundColour(self._theme.bar_bg)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        def _lbl(text: str) -> wx.StaticText:
            w = wx.StaticText(bar, label=text)
            w.SetFont(_ui_font(9))
            w.SetBackgroundColour(self._theme.bar_bg)
            w.SetForegroundColour(self._theme.bar_fg)
            return w

        rec_codes = _REC_LANG_CODES
        rec_labels = [
            self._t("menu_lang_pt"),
            self._t("menu_lang_en"),
            self._t("menu_rec_auto"),
        ]
        self._rec_lang_choice = wx.Choice(bar, choices=rec_labels)
        self._rec_lang_choice.SetSelection(rec_codes.index(self._rec_lang))
        self._rec_lang_choice.SetFont(_ui_font(9))

        ui_labels = [self._t("menu_lang_pt"), self._t("menu_lang_en")]
        self._ui_lang_choice = wx.Choice(bar, choices=ui_labels)
        self._ui_lang_choice.SetSelection(_UI_LANG_CODES.index(self._ui_lang))
        self._ui_lang_choice.SetFont(_ui_font(9))

        self._conf_badge = _ConfidenceBadge(bar, self._theme)

        theme_btn = wx.Button(bar, label="◑", size=(32, 26))
        theme_btn.SetFont(_ui_font(11, bold=True))
        theme_btn.SetToolTip(self._t("btn_theme_tip"))
        theme_btn.SetBackgroundColour(self._theme.sec_btn_bg)
        theme_btn.SetForegroundColour(self._theme.sec_btn_fg)
        theme_btn.Bind(wx.EVT_BUTTON, lambda _: self._toggle_theme())

        sizer.Add(_lbl(self._t("lbl_trans_lang")), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
        sizer.Add(self._rec_lang_choice, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP | wx.BOTTOM, 6)
        sizer.Add(_lbl(self._t("lbl_ui_lang")), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 14)
        sizer.Add(self._ui_lang_choice, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP | wx.BOTTOM, 6)
        sizer.AddStretchSpacer()
        sizer.Add(_lbl(self._t("lbl_confidence")), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        sizer.Add(self._conf_badge, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        sizer.Add(theme_btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT | wx.TOP | wx.BOTTOM, 6)

        bar.SetSizer(sizer)
        bar.SetMinSize((-1, 40))
        return bar

    def _build_btn_bar(self, parent: wx.Window) -> wx.Panel:
        bar = wx.Panel(parent)
        bar.SetBackgroundColour(self._theme.bg)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        buttons = [
            ("btn_save", self._on_save),
            ("btn_open", self._on_open),
            ("btn_clear", self._on_clear),
            ("btn_copy", self._on_copy),
        ]

        btn_attrs = ["_btn_save", "_btn_open", "_btn_clear", "_btn_copy"]

        for (key, handler), attr in zip(buttons, btn_attrs):
            btn = wx.Button(bar, label=self._t(key))
            btn.SetFont(_ui_font(10))
            btn.SetBackgroundColour(self._theme.sec_btn_bg)
            btn.SetForegroundColour(self._theme.sec_btn_fg)
            btn.Bind(wx.EVT_BUTTON, handler)
            setattr(self, attr, btn)
            sizer.Add(btn, 0, wx.RIGHT, 8)

        bar.SetSizer(sizer)
        return bar

    def _build_rec_row(self, parent: wx.Window) -> wx.Panel:
        row = wx.Panel(parent)
        row.SetBackgroundColour(self._theme.bg)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self._vad_dot = _VadDot(row)
        self._vad_dot.set_colour(self._theme.vad_idle)

        self._rec_btn = wx.Button(row, label=self._t("btn_record"))
        self._rec_btn.SetFont(_ui_font(13, bold=True))
        self._rec_btn.SetBackgroundColour(self._theme.primary)
        self._rec_btn.SetForegroundColour(self._theme.primary_fg)
        self._rec_btn.SetMinSize((180, 46))
        self._rec_btn.Disable()
        self._rec_btn.Bind(wx.EVT_BUTTON, lambda _: self._toggle_recording())

        sizer.AddStretchSpacer()
        sizer.Add(self._vad_dot, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        sizer.Add(self._rec_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.AddStretchSpacer()

        row.SetSizer(sizer)
        row.SetMinSize((-1, 60))
        return row

    def _build_menu(self) -> None:
        bar = wx.MenuBar()
        t = self._t

        file_menu = wx.Menu()
        file_menu.Append(wx.ID_NEW, t("menu_new"))
        file_menu.Append(wx.ID_OPEN, t("menu_open"))
        file_menu.Append(wx.ID_SAVE, t("menu_save"))
        file_menu.Append(wx.ID_SAVEAS, t("menu_save_as"))
        file_menu.AppendSeparator()
        file_menu.Append(wx.ID_EXIT, t("menu_exit"))

        view_menu = wx.Menu()
        view_menu.Append(_ID_TOGGLE_THEME, t("menu_theme"))
        view_menu.AppendSeparator()
        self._mi_word_wrap = view_menu.AppendCheckItem(_ID_WORD_WRAP, t("menu_word_wrap"))
        self._mi_word_wrap.Check(self._word_wrap)
        view_menu.Append(_ID_FONT_INC, t("menu_font_inc"))
        view_menu.Append(_ID_FONT_DEC, t("menu_font_dec"))

        ui_lang_menu = wx.Menu()
        self._mi_ui_pt = ui_lang_menu.AppendRadioItem(_ID_UI_PT, t("menu_lang_pt"))
        self._mi_ui_en = ui_lang_menu.AppendRadioItem(_ID_UI_EN, t("menu_lang_en"))
        self._mi_ui_pt.Check(self._ui_lang == "pt")
        self._mi_ui_en.Check(self._ui_lang == "en")
        view_menu.AppendSubMenu(ui_lang_menu, t("menu_ui_lang"))

        rec_lang_menu = wx.Menu()
        self._mi_rec_pt = rec_lang_menu.AppendRadioItem(_ID_REC_PT, t("menu_lang_pt"))
        self._mi_rec_en = rec_lang_menu.AppendRadioItem(_ID_REC_EN, t("menu_lang_en"))
        self._mi_rec_auto = rec_lang_menu.AppendRadioItem(_ID_REC_AUTO, t("menu_rec_auto"))
        self._mi_rec_pt.Check(True)
        view_menu.AppendSubMenu(rec_lang_menu, t("menu_rec_lang"))

        bar.Append(file_menu, t("menu_file"))
        bar.Append(view_menu, t("menu_view"))
        self.SetMenuBar(bar)

    def _bind_events(self) -> None:
        self.Bind(wx.EVT_MENU, lambda _: self._new_note(), id=wx.ID_NEW)
        self.Bind(wx.EVT_MENU, lambda _: self._on_open(), id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, lambda _: self._on_save(), id=wx.ID_SAVE)
        self.Bind(wx.EVT_MENU, lambda _: self._on_save_as(), id=wx.ID_SAVEAS)
        self.Bind(wx.EVT_MENU, lambda _: self.Close(), id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, lambda _: self._toggle_theme(), id=_ID_TOGGLE_THEME)
        self.Bind(wx.EVT_MENU, lambda _: self._toggle_word_wrap(), id=_ID_WORD_WRAP)
        self.Bind(wx.EVT_MENU, lambda _: self._change_font_size(1), id=_ID_FONT_INC)
        self.Bind(wx.EVT_MENU, lambda _: self._change_font_size(-1), id=_ID_FONT_DEC)
        self.Bind(wx.EVT_MENU, self._on_menu_ui_pt, id=_ID_UI_PT)
        self.Bind(wx.EVT_MENU, self._on_menu_ui_en, id=_ID_UI_EN)
        self.Bind(wx.EVT_MENU, self._on_menu_rec_pt, id=_ID_REC_PT)
        self.Bind(wx.EVT_MENU, self._on_menu_rec_en, id=_ID_REC_EN)
        self.Bind(wx.EVT_MENU, self._on_menu_rec_auto, id=_ID_REC_AUTO)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_key)
        if self._editor:
            self._editor.Bind(wx.EVT_TEXT, self._on_text_change)
        if self._rec_lang_choice:
            self._rec_lang_choice.Bind(wx.EVT_CHOICE, self._on_rec_lang_choice)
        if self._ui_lang_choice:
            self._ui_lang_choice.Bind(wx.EVT_CHOICE, self._on_ui_lang_choice)

    def _start_model_load(self) -> None:
        threading.Thread(
            target=self._load_model_thread,
            daemon=True,
            name="model-loader",
        ).start()

    def _load_model_thread(self) -> None:
        try:
            self._transcriber.load(_MODEL_SIZE)
            self._engine = AudioEngine(
                on_utterance=self._on_utterance,
                on_vad_change=self._on_vad_change,
            )
            self._worker = TranscriptionWorker(
                transcriber=self._transcriber,
                on_result=self._on_transcription_result,
                on_error=self._on_transcription_error,
                on_idle_change=self._on_worker_idle,
            )
            self._worker.start()
            wx.CallAfter(self._on_model_ready)
        except Exception as exc:
            logger.exception("Model load failure")
            wx.CallAfter(self._on_model_error, exc)

    def _on_model_ready(self) -> None:
        self._model_ready = True
        if self._loading_panel:
            self._loading_panel.stop_animation()
            self._loading_panel.Hide()
        if self._editor_panel:
            self._editor_panel.Show()
        if self._rec_btn:
            self._rec_btn.Enable()
        self.Layout()
        logger.info("Model ready.")

    def _on_model_error(self, exc: Exception) -> None:
        if self._loading_panel:
            self._loading_panel.show_error(exc)
            self._loading_panel.Layout()

    def _toggle_recording(self) -> None:
        if not self._model_ready:
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        try:
            if self._engine is None:
                raise RuntimeError("Audio engine not initialized.")
            self._engine.start()
        except Exception as exc:
            logger.exception("Audio engine start failed")
            self._flash_status(f"{self._t('msg_mic_error')}: {safe_error_message(exc)}")
            return
        self._recording = True
        if self._rec_btn:
            self._rec_btn.SetLabel(self._t("btn_stop"))
            self._rec_btn.SetBackgroundColour(self._theme.stop_bg)
            self._rec_btn.SetForegroundColour(self._theme.stop_fg)
            self._rec_btn.Refresh()
        if self._vad_dot:
            self._vad_dot.set_colour(self._theme.vad_listening)
        self._set_status(self._t("recording"))
        self._start_pulse()

    def _stop_recording(self) -> None:
        self._recording = False
        self._stop_pulse()
        if self._engine:
            self._engine.stop()
        if self._rec_btn:
            self._rec_btn.SetLabel(self._t("btn_record"))
            self._rec_btn.SetBackgroundColour(self._theme.primary)
            self._rec_btn.SetForegroundColour(self._theme.primary_fg)
            self._rec_btn.Refresh()
        if self._vad_dot:
            self._vad_dot.set_colour(self._theme.vad_idle)
        self._set_status(self._t("ready"))

    def _on_utterance(self, audio: np.ndarray) -> None:
        if self._worker:
            self._worker.submit(audio)

    def _on_vad_change(self, active: bool) -> None:
        if self._recording and self._vad_dot:
            colour = self._theme.vad_speaking if active else self._theme.vad_listening
            wx.CallAfter(self._vad_dot.set_colour, colour)

    def _on_transcription_result(
        self, text: str, conf: float, lang: str
    ) -> None:
        self._last_conf = conf
        wx.CallAfter(self._append_transcription, text, conf)

    def _append_transcription(self, text: str, conf: float = 0.0) -> None:
        if not self._editor or not text:
            return

        text = sanitize_transcription(text)
        if not text:
            return

        if check_prompt_injection(text):
            self._injection_warnings += 1
            logger.warning(
                "Prompt injection in transcription output (count=%d)", self._injection_warnings
            )
            wx.CallAfter(self._flash_status, self._t("msg_injection_warning"))
            return

        current = self._editor.GetValue()
        if len(current) + len(text) + 1 > _MAX_EDITOR_CHARS:
            self._flash_status(self._t("msg_too_large"))
            return

        sep = " " if current and current[-1] not in {"\n", " "} else ""
        safe = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        self._editor.AppendText(sep + safe)
        self._editor.SetInsertionPointEnd()
        self._unsaved = True
        self._update_counts()

        if self._conf_badge:
            self._conf_badge.update(conf, self._theme)

    def _on_transcription_error(self, msg: str) -> None:
        logger.error("Transcription error: %s", msg)
        safe_msg = safe_error_message(Exception(msg))
        wx.CallAfter(self._flash_status, f"{self._t('msg_error')}: {safe_msg}")

    def _on_worker_idle(self, idle: bool) -> None:
        if self._recording:
            label = self._t("recording") if idle else self._t("processing")
            wx.CallAfter(self._set_status, label)

    def _on_text_change(self, _: wx.CommandEvent) -> None:
        self._unsaved = True
        self._update_counts()

    def _update_counts(self) -> None:
        if not self._editor or not self._status_bar:
            return
        txt = self._editor.GetValue()
        words = len(txt.split()) if txt.strip() else 0
        self._status_bar.SetStatusText(
            f"{words} {self._t('words')}  ·  {len(txt)} {self._t('chars')}",
            2,
        )

    def _set_status(self, text: str) -> None:
        if self._status_bar:
            self._status_bar.SetStatusText(text, 0)

    def _flash_status(self, text: str) -> None:
        self._set_status(text)
        wx.CallLater(_FLASH_MS, self._set_status, self._t("ready"))

    def _new_note(self) -> None:
        if self._unsaved and self._editor and self._editor.GetValue().strip():
            dlg = wx.MessageDialog(
                self,
                message=self._t("warn_unsaved"),
                caption=self._t("warn_unsaved_title"),
                style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            )
            result = dlg.ShowModal()
            dlg.Destroy()
            if result != wx.ID_YES:
                return
        if self._editor:
            self._editor.SetValue("")
        self._unsaved = False
        self._transcriber.clear_context()
        self._file_manager.new()
        if self._status_bar:
            self._status_bar.SetStatusText(self._t("lbl_untitled"), 1)
        if self._conf_badge:
            self._conf_badge.hide_badge()
        self._update_counts()

    def _on_save(self, _: Optional[wx.CommandEvent] = None) -> None:
        if self._file_manager.current_path:
            self._write_current()
        else:
            self._on_save_as()

    def _on_save_as(self, _: Optional[wx.CommandEvent] = None) -> None:
        dlg = wx.FileDialog(
            self,
            message=self._t("dlg_save_title"),
            defaultFile="note.txt",
            wildcard=self._t("file_filter"),
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        if dlg.ShowModal() == wx.ID_OK:
            raw = dlg.GetPath()
            dest = raw if raw.lower().endswith((".txt", ".md")) else raw + ".txt"
            try:
                content = self._editor.GetValue() if self._editor else ""
                self._file_manager.save(content, filepath=dest)
                self._unsaved = False
                if self._status_bar and self._file_manager.current_path:
                    self._status_bar.SetStatusText(
                        str(self._file_manager.current_path.name), 1
                    )
                self._flash_status(self._t("msg_saved"))
            except Exception as exc:
                self._flash_status(f"{self._t('msg_error')}: {safe_error_message(exc)}")
        dlg.Destroy()

    def _write_current(self) -> None:
        try:
            content = self._editor.GetValue() if self._editor else ""
            self._file_manager.save(content)
            self._unsaved = False
            self._flash_status(self._t("msg_saved"))
        except Exception as exc:
            self._flash_status(f"{self._t('msg_error')}: {safe_error_message(exc)}")

    def _on_open(self, _: Optional[wx.CommandEvent] = None) -> None:
        dlg = wx.FileDialog(
            self,
            message=self._t("dlg_open_title"),
            wildcard=self._t("file_filter"),
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                content = self._file_manager.load(dlg.GetPath())
                if self._editor:
                    self._editor.SetValue(content)
                self._unsaved = False
                if self._status_bar and self._file_manager.current_path:
                    self._status_bar.SetStatusText(
                        str(self._file_manager.current_path.name), 1
                    )
                self._update_counts()
            except Exception as exc:
                self._flash_status(f"{self._t('msg_error')}: {safe_error_message(exc)}")
        dlg.Destroy()

    def _on_clear(self, _: Optional[wx.CommandEvent] = None) -> None:
        dlg = wx.MessageDialog(
            self,
            message=self._t("dlg_clear_body"),
            caption=self._t("dlg_clear_title"),
            style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
        )
        if dlg.ShowModal() == wx.ID_YES and self._editor:
            self._editor.SetValue("")
            self._transcriber.clear_context()
            self._unsaved = True
            if self._conf_badge:
                self._conf_badge.hide_badge()
            self._update_counts()
        dlg.Destroy()

    def _on_copy(self, _: Optional[wx.CommandEvent] = None) -> None:
        if not self._editor:
            return
        txt = self._editor.GetValue()
        if not txt.strip():
            self._flash_status(self._t("msg_no_text"))
            return
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(txt))
            wx.TheClipboard.Close()
            self._flash_status(self._t("msg_copied"))

    def _toggle_theme(self) -> None:
        self._theme = _DARK if self._theme is _LIGHT else _LIGHT
        self._apply_theme()

    def _toggle_word_wrap(self) -> None:
        self._word_wrap = not self._word_wrap
        if self._mi_word_wrap:
            self._mi_word_wrap.Check(self._word_wrap)
        if self._editor:
            current_text = self._editor.GetValue()
            cursor_pos = self._editor.GetInsertionPoint()
            self._editor.Destroy()
            self._editor = None

            editor_style = wx.TE_MULTILINE | wx.BORDER_NONE
            if self._word_wrap:
                editor_style |= wx.TE_WORDWRAP
            else:
                editor_style |= wx.TE_DONTWRAP

            if self._editor_panel:
                sizer = self._editor_panel.GetSizer()
                self._editor = wx.TextCtrl(
                    self._editor_panel,
                    style=editor_style,
                )
                self._editor.SetFont(_ui_font(self._font_size))
                self._editor.SetBackgroundColour(self._theme.input_bg)
                self._editor.SetForegroundColour(self._theme.input_fg)
                self._editor.SetHint(self._t("hint"))
                self._editor.SetMaxLength(0)
                self._editor.SetValue(current_text)
                self._editor.SetInsertionPoint(min(cursor_pos, len(current_text)))
                self._editor.Bind(wx.EVT_TEXT, self._on_text_change)
                if sizer:
                    sizer.Insert(2, self._editor, 1, wx.EXPAND | wx.ALL, 12)
                self._editor_panel.Layout()

    def _change_font_size(self, delta: int) -> None:
        new_size = max(_FONT_SIZE_MIN, min(_FONT_SIZE_MAX, self._font_size + delta))
        if new_size == self._font_size:
            return
        self._font_size = new_size
        if self._editor:
            self._editor.SetFont(_ui_font(self._font_size))
            self._editor.Refresh()

    def _apply_theme(self) -> None:
        t = self._theme

        def _recolour(w: wx.Window, bg: wx.Colour, fg: wx.Colour) -> None:
            w.SetBackgroundColour(bg)
            w.SetForegroundColour(fg)

        if self._editor_panel:
            _recolour(self._editor_panel, t.bg, t.fg)

        if self._editor:
            _recolour(self._editor, t.input_bg, t.input_fg)

        if self._loading_panel:
            self._loading_panel._apply_theme(t)

        if self._rec_btn:
            if self._recording:
                _recolour(self._rec_btn, t.stop_bg, t.stop_fg)
            else:
                _recolour(self._rec_btn, t.primary, t.primary_fg)

        if self._vad_dot:
            colour = (
                t.vad_speaking if self._recording else t.vad_idle
            )
            self._vad_dot.set_colour(colour)

        if self._conf_badge:
            self._conf_badge._theme = t
            self._conf_badge.Refresh()

        if self._status_bar:
            _recolour(self._status_bar, t.bar_bg, t.bar_fg)

        for attr in ("_btn_save", "_btn_open", "_btn_clear", "_btn_copy"):
            btn = getattr(self, attr, None)
            if btn:
                _recolour(btn, t.sec_btn_bg, t.sec_btn_fg)
                btn.Refresh()

        if self._editor_panel:
            for panel in self._editor_panel.GetChildren():
                if isinstance(panel, wx.Panel):
                    bg = t.bar_bg if panel is self._get_lang_bar() else t.bg
                    panel.SetBackgroundColour(bg)
                    for child in panel.GetChildren():
                        if isinstance(child, wx.StaticText):
                            child.SetBackgroundColour(bg)
                            child.SetForegroundColour(t.bar_fg if bg == t.bar_bg else t.hint_fg)

        if self._rec_lang_choice:
            self._rec_lang_choice.SetBackgroundColour(self._theme.bar_bg)
        if self._ui_lang_choice:
            self._ui_lang_choice.SetBackgroundColour(self._theme.bar_bg)

        self.SetBackgroundColour(t.bg)
        self.Refresh()
        self.Layout()

    def _get_lang_bar(self) -> Optional[wx.Panel]:
        if not self._editor_panel:
            return None
        for child in self._editor_panel.GetChildren():
            if isinstance(child, wx.Panel) and child.GetMinHeight() == 40:
                return child
        return None

    def _start_pulse(self) -> None:
        self._pulse_on = True
        self._pulse_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_pulse_tick, self._pulse_timer)
        self._pulse_timer.Start(_PULSE_MS)

    def _on_pulse_tick(self, _: wx.TimerEvent) -> None:
        if not self._recording or not self._vad_dot:
            return
        colour = self._theme.vad_listening if self._pulse_on else self._theme.vad_idle
        self._vad_dot.set_colour(colour)
        self._pulse_on = not self._pulse_on

    def _stop_pulse(self) -> None:
        if self._pulse_timer and self._pulse_timer.IsRunning():
            self._pulse_timer.Stop()
        self._pulse_timer = None

    def _on_key(self, event: wx.KeyEvent) -> None:
        if event.ControlDown():
            k = event.GetKeyCode()
            if k == ord("R"):
                self._toggle_recording()
                return
            if k == ord("S"):
                self._on_save()
                return
            if k == ord("O"):
                self._on_open()
                return
            if k == ord("=") or k == wx.WXK_NUMPAD_ADD:
                self._change_font_size(1)
                return
            if k == ord("-") or k == wx.WXK_NUMPAD_SUBTRACT:
                self._change_font_size(-1)
                return
        event.Skip()

    def _on_rec_lang_choice(self, event: wx.CommandEvent) -> None:
        idx = event.GetSelection()
        if 0 <= idx < len(_REC_LANG_CODES):
            self._rec_lang = _REC_LANG_CODES[idx]
            self._transcriber.language = self._rec_lang
            self._sync_rec_lang_menu()

    def _on_ui_lang_choice(self, event: wx.CommandEvent) -> None:
        idx = event.GetSelection()
        if 0 <= idx < len(_UI_LANG_CODES):
            self._ui_lang = _UI_LANG_CODES[idx]
            self._refresh_strings()

    def _on_menu_rec_pt(self, _: wx.CommandEvent) -> None:
        self._set_rec_lang("pt")

    def _on_menu_rec_en(self, _: wx.CommandEvent) -> None:
        self._set_rec_lang("en")

    def _on_menu_rec_auto(self, _: wx.CommandEvent) -> None:
        self._set_rec_lang("auto")

    def _on_menu_ui_pt(self, _: wx.CommandEvent) -> None:
        self._ui_lang = "pt"
        self._refresh_strings()

    def _on_menu_ui_en(self, _: wx.CommandEvent) -> None:
        self._ui_lang = "en"
        self._refresh_strings()

    def _set_rec_lang(self, lang: str) -> None:
        self._rec_lang = lang
        self._transcriber.language = lang
        if self._rec_lang_choice and lang in _REC_LANG_CODES:
            self._rec_lang_choice.SetSelection(_REC_LANG_CODES.index(lang))
        self._sync_rec_lang_menu()

    def _sync_rec_lang_menu(self) -> None:
        mapping = {"pt": self._mi_rec_pt, "en": self._mi_rec_en, "auto": self._mi_rec_auto}
        item = mapping.get(self._rec_lang)
        if item:
            item.Check(True)

    def _refresh_strings(self) -> None:
        self.SetTitle(self._t("title"))
        self._set_status(self._t("ready"))
        if self._rec_btn and not self._recording:
            self._rec_btn.SetLabel(self._t("btn_record"))
        for attr, key in [
            ("_btn_save", "btn_save"),
            ("_btn_open", "btn_open"),
            ("_btn_clear", "btn_clear"),
            ("_btn_copy", "btn_copy"),
        ]:
            btn = getattr(self, attr, None)
            if btn:
                btn.SetLabel(self._t(key))
        if self._ui_lang_choice and self._ui_lang in _UI_LANG_CODES:
            self._ui_lang_choice.SetSelection(_UI_LANG_CODES.index(self._ui_lang))
        self._update_counts()
        self.Refresh()
        self.Layout()

    def _on_close(self, event: wx.CloseEvent) -> None:
        if self._unsaved and self._editor and self._editor.GetValue().strip():
            dlg = wx.MessageDialog(
                self,
                message=self._t("warn_unsaved"),
                caption=self._t("warn_unsaved_title"),
                style=wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            )
            result = dlg.ShowModal()
            dlg.Destroy()
            if result != wx.ID_YES:
                event.Veto()
                return
        self._cleanup()
        event.Skip()

    def _cleanup(self) -> None:
        try:
            self._stop_pulse()
            if self._recording and self._engine:
                self._engine.stop()
            if self._worker:
                self._worker.stop()
        except Exception:
            logger.debug("Cleanup error (non-fatal)", exc_info=True)
        logger.info("Cleanup complete.")


class VoiceNotepadApp(wx.App):
    def OnInit(self) -> bool:
        frame = MainFrame()
        frame.Show()
        self.SetTopWindow(frame)
        return True
