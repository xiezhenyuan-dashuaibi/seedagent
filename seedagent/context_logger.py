from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = AGENT_DIR / "logs"
DEFAULT_MODEL_INPUT_LOG_FILE = DEFAULT_LOG_DIR / "model_inputs.md"


def _safe_json(value: Any) -> Any:
    if is_dataclass(value):
        return _safe_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _safe_json(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _safe_json(vars(value))
        except Exception:
            pass
    return repr(value)


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _drop_empty(item)
            for key, item in value.items()
            if item is not None and item != [] and item != {}
        }
        return {
            key: item
            for key, item in cleaned.items()
            if item is not None and item != [] and item != {}
        }
    if isinstance(value, list):
        return [_drop_empty(item) for item in value]
    return value


def _messages_to_dict(messages: list[Any] | None) -> list[dict[str, Any]]:
    if not messages:
        return []
    out = []
    for message in messages:
        value = _safe_json(message)
        cleaned = _drop_empty(value)
        out.append(cleaned if isinstance(cleaned, dict) else {"value": cleaned})
    return out


class ModelInputLogger:
    """Overwrite-only snapshot of the latest messages sent into model.generate()."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        run_name: str | None = None,
        *,
        write_jsonl: bool = False,
        log_file: str | Path | None = None,
        jsonl_file: str | Path | None = None,
    ):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_path = Path(log_file) if log_file else self.log_dir / DEFAULT_MODEL_INPUT_LOG_FILE.name
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = None

    def log_model_input(
        self,
        messages: list[Any],
        *,
        model_id: str | None = None,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tools_to_call_from: list[Any] | None = None,
    ) -> None:
        self.markdown_path.write_text(
            json.dumps(_messages_to_dict(messages), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_MODEL_INPUT_LOG_FILE",
    "ModelInputLogger",
]
