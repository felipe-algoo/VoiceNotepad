import logging
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_LOG_PATH = _ROOT / "voice_notepad.log"

try:
    _file_handler = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8", delay=False)
except (OSError, PermissionError):
    _file_handler = logging.NullHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr), _file_handler],
)

import uvicorn

from src.app import app


def main() -> None:
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="warning",
        ws_max_size=4 * 1024 * 1024,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()