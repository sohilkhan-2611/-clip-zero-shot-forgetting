"""Zero-shot evaluation: accuracy, cross-entropy loss and correct-class margin.

Run:  python -m src.evaluate --config configs/eval_zeroshot_vitb16.yaml [--set key=value ...]

Works for the untouched model and, with `checkpoint:` set, for a fine-tuned one
(the checkpoint holds only the trained tensors, see src/finetune.py).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from src import data as D
from src import model as M
from src.utils import get_device, load_config, results_path, run_metadata, save_json, set_seed

# our dataset name -> column in open_clip's docs/openclip_results.csv
REFERENCE_COLUMNS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "dtd": "Describable Textures",
    "food101": "Food-101",
    "flowers": "Oxford Flowers-102",
    "eurosat": "EuroSAT",
    "pets": "Oxford-IIIT Pet",
}


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Accuracy, mean cross-entropy, mean correct-class margin.

    margin_i = logit[i, y_i] - max_{j != y_i} logit[i, j]  (positive means correct, in logit units).
    """
    logits = logits.float()
    targets = targets.long()
    acc = (logits.argmax(dim=1) == targets).float().mean().item()
    loss = F.cross_entropy(logits, targets).item()
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    others = logits.scatter(1, targets[:, None], float("-inf"))
    margin = (correct - others.max(dim=1).values).mean().item()
    return {"accuracy": acc, "loss": loss, "margin": margin, "n": int(targets.numel())}


def _autocast(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_logits(model, loader, anchors, device, amp: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """All logits and targets of a loader, on CPU in float32."""
    model.eval()
    all_logits, all_targets = [], []
    for images, targets in loader:
        with _autocast(device, amp):
            logits = M.image_logits(model, images.to(device, non_blocking=True), anchors.to(device))
        all_logits.append(logits.float().cpu())
        all_targets.append(torch.as_tensor(targets))
    return torch.cat(all_logits), torch.cat(all_targets)


def evaluate_dataset(model, tokenizer, ds, device, batch_size: int, num_workers: int, amp: bool = True) -> dict:
    anchors = M.build_text_anchors(model, tokenizer, ds.classes, ds.templates, device=device)
    loader = D.make_loader(ds, batch_size, num_workers, shuffle=False)
    logits, targets = predict_logits(model, loader, anchors, device, amp)
    out = compute_metrics(logits, targets)
    out["num_classes"] = len(ds.classes)
    out["num_templates"] = len(ds.templates)
    return out


def load_reference(url: str, model_name: str, pretrained: str, cache: str | Path) -> dict | None:
    """Published OpenCLIP zero-shot accuracies for one (model, pretrained) row, keyed by our dataset names.

    Downloaded at run time from the open_clip repo and cached; never typed in by hand.
    Returns None if the file cannot be fetched or the row is missing.
    """
    cache = Path(cache)
    try:
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=30) as r:
                cache.write_bytes(r.read())
        rows = list(csv.DictReader(io.StringIO(cache.read_text())))
    except Exception as e:  # network down etc.
        print(f"[reference] could not load {url}: {e}")
        return None
    row = next((r for r in rows if r["name"] == model_name and r["pretrained"] == pretrained), None)
    if row is None:
        return None
    return {ours: float(row[col]) for ours, col in REFERENCE_COLUMNS.items() if row.get(col)}


def compare_to_reference(results: dict, reference: dict | None, tolerance_pp: float) -> dict:
    """Per dataset: ours, published, difference in percentage points, and whether it is within tolerance."""
    out = {}
    for name, res in results.items():
        pub = (reference or {}).get(name)
        if pub is None:
            out[name] = {"ours": res["accuracy"], "published": None, "diff_pp": None, "within_tolerance": None}
            continue
        diff = 100.0 * (res["accuracy"] - pub)
        out[name] = {"ours": res["accuracy"], "published": pub, "diff_pp": diff,
                     "within_tolerance": abs(diff) <= tolerance_pp}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like data.max_samples=256")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    mcfg, dcfg = cfg["model"], cfg["data"]

    model, _, preprocess_val, tokenizer = M.load_clip(mcfg["name"], mcfg["pretrained"], device)
    if cfg.get("checkpoint"):
        M.load_partial_state_dict(model, torch.load(cfg["checkpoint"], map_location="cpu"))

    results = {}
    for name in dcfg["eval_sets"]:
        ds = D.build_dataset(name, "test", dcfg["root"], preprocess_val,
                             split_seed=dcfg.get("split_seed", 0),
                             eurosat_test_fraction=dcfg.get("eurosat_test_fraction", 0.2))
        ds = D.subsample(ds, dcfg.get("max_samples"), cfg["seed"])
        results[name] = evaluate_dataset(model, tokenizer, ds, device, dcfg["batch_size"],
                                         dcfg["num_workers"], cfg.get("amp", True))
        print(f"{name:10s} acc={results[name]['accuracy']:.4f} loss={results[name]['loss']:.4f} "
              f"margin={results[name]['margin']:.3f} n={results[name]['n']}")

    out = {"config": cfg, "meta": run_metadata(), "results": results}
    rcfg = cfg.get("reference")
    if rcfg and not cfg.get("checkpoint"):
        ref = load_reference(rcfg["url"], mcfg["name"], mcfg["pretrained"],
                             Path(dcfg["root"]) / "openclip_results.csv")
        out["reference_comparison"] = compare_to_reference(results, ref, rcfg.get("tolerance_pp", 1.0))
        out["reference_note"] = ("Published values come from open_clip docs/openclip_results.csv "
                                 "(clip_benchmark). EuroSAT uses our own split, so compare loosely.")
        for name, c in out["reference_comparison"].items():
            if c["published"] is not None:
                print(f"{name:10s} ours={c['ours']:.4f} published={c['published']:.4f} diff={c['diff_pp']:+.2f}pp")
    path = save_json(out, results_path(cfg))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
