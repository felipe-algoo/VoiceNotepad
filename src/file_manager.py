import logging
import pathlib
from typing import Optional

from src.security import validate_path, validate_content

logger = logging.getLogger(__name__)


class FileManager:
    def __init__(self) -> None:
        self._current: Optional[pathlib.Path] = None
        self._default_dir: pathlib.Path = pathlib.Path.home() / "Documents"
        self._default_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Optional[pathlib.Path]:
        return self._current

    @property
    def default_dir(self) -> pathlib.Path:
        return self._default_dir

    def new(self) -> None:
        self._current = None

    def save(self, content: str, filepath: Optional[str] = None) -> pathlib.Path:
        validate_content(content)
        if filepath:
            target = validate_path(filepath)
        elif self._current:
            target = self._current
        else:
            target = self._default_dir / "untitled.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._current = target
        logger.info("Saved: %s", target)
        return target

    def load(self, filepath: str) -> str:
        target = validate_path(filepath)
        if not target.exists():
            raise FileNotFoundError(f"File not found.")
        if not target.is_file():
            raise ValueError(f"Path is not a regular file.")
        content = target.read_text(encoding="utf-8")
        self._current = target
        logger.info("Loaded: %s", target)
        return content
