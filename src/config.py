"""Loads config.yaml once and exposes project paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(key: str) -> Path:
    """Absolute path for a key under `paths:` in config.yaml (created if missing)."""
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
