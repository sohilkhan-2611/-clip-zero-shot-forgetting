"""Helpers for running experiments on Colab safely: checkpoint checks, result syncing, shell commands.

Nothing here changes what is trained or evaluated; it only decides whether to run a step
and where result files are kept.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# Trained tensors per transformer block, as named by open_clip 3.3.0 ViT-B-16
# (checked against model.freeze_for_finetune in tests/test_runner.py).
BLOCK_PARAMS = (
    "ln_1.weight", "ln_1.bias",
    "attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias",
    "ln_2.weight", "ln_2.bias",
    "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias",
)


def expected_trainable_keys(train_last_k_blocks: int, n_blocks: int = 12) -> set[str]:
    """Names of the tensors a fine-tune of the last k visual blocks saves."""
    return {f"visual.transformer.resblocks.{i}.{p}"
            for i in range(n_blocks - train_last_k_blocks, n_blocks) for p in BLOCK_PARAMS}


def checkpoint_status(path: str | os.PathLike, expected_keys: set[str]) -> tuple[bool, str]:
    """(valid, reason). Valid means: file loads, holds exactly the expected tensors, all finite."""
    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as e:  # truncated or corrupt file
        return False, f"unreadable ({type(e).__name__})"
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        return False, "not a dict of tensors"
    if set(state) != expected_keys:
        return False, f"wrong tensors ({len(state)} found, {len(expected_keys)} expected)"
    bad = [k for k, v in state.items() if not torch.isfinite(v).all()]
    if bad:
        return False, f"non-finite values in {len(bad)} tensors"
    return True, f"valid ({len(state)} tensors)"


def should_train(path: str | os.PathLike, expected_keys: set[str]) -> bool:
    """True if training is needed. A broken checkpoint is renamed to *.invalid, never silently overwritten."""
    ok, reason = checkpoint_status(path, expected_keys)
    if ok:
        print(f"skip training: {path} is {reason}")
        return False
    if Path(path).exists():
        moved = Path(str(path) + ".invalid")
        Path(path).rename(moved)
        print(f"checkpoint {path} is {reason}; moved to {moved}, training again")
    else:
        print(f"no checkpoint at {path}; training")
    return True


def sync_json(src_dir: str | os.PathLike, dst_dir: str | os.PathLike) -> list[str]:
    """Copy *.json from src to dst when missing in dst or newer in src. Returns copied file names."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(src_dir.glob("*.json")):
        target = dst_dir / f.name
        if not target.exists() or f.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(f, target)
            copied.append(f.name)
    return copied


def run(cmd: str) -> None:
    """Run a shell command, stream its output live, and raise if it fails (so 'Run all' stops)."""
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if proc.wait() != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd}")
