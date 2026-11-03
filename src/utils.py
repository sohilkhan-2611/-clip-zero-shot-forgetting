"""Shared helpers: config loading, seeding, device choice, JSON results with run metadata."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _parse_value(text: str) -> Any:
    """Parse a CLI override value with YAML rules, so '3' -> 3, 'null' -> None, '[a,b]' -> list.

    PyYAML reads '1e-5' as a string (YAML 1.1 quirk), so numeric-looking strings become floats.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply 'a.b.c=value' overrides in place and return cfg."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Read a YAML config and apply optional CLI overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(path)
    return apply_overrides(cfg, overrides)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fix python, numpy and torch seeds. deterministic=True also forces deterministic cuDNN (slower)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:
        return None


def run_metadata() -> dict:
    """Everything needed to tell later where a number came from."""
    import open_clip  # local import keeps utils importable without open_clip

    meta = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "open_clip": open_clip.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
    }
    try:
        from importlib.metadata import version

        meta["clip_benchmark"] = version("clip_benchmark")
    except Exception:
        meta["clip_benchmark"] = None
    return meta


def results_path(cfg: dict, suffix: str = "") -> Path:
    """results/<phase dir>/<experiment>[_suffix]_<UTC time>.json so runs never overwrite each other."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = cfg.get("experiment", "run") + (f"_{suffix}" if suffix else "") + f"_{stamp}.json"
    return Path(cfg.get("output", {}).get("results_dir", "results")) / name


def save_json(obj: Any, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")
