from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> Path:
    """Create a writable directory, falling back when cloud-synced folders reject writes."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        _assert_writable(path)
        return path
    except OSError as exc:
        fallback = _fallback_path(path)
        fallback.mkdir(parents=True, exist_ok=True)
        _assert_writable(fallback)
        warnings.warn(
            f"Could not write to '{path}' ({exc}). Using '{fallback}' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback


def _assert_writable(path: Path) -> None:
    probe = path / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _fallback_path(path: Path) -> Path:
    root = Path(
        os.environ.get(
            "EXPLAINABLE_PDM_OUTPUT_DIR",
            Path(tempfile.gettempdir()) / "explainable_predictive_maintenance",
        )
    )
    try:
        suffix = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        suffix = Path(path.name)
    return root / suffix


def model_path_candidates(project_model_path: Path) -> list[Path]:
    return [
        Path(project_model_path),
        _fallback_path(Path(project_model_path)),
    ]


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(model: Any, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)
