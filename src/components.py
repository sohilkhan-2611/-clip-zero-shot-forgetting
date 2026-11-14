"""Phase 1: where does forgetting live? Revert / retain component updates of a fine-tuned CLIP.

Run:  python -m src.components --config configs/phase1_components_flowers.yaml [--set ...]

Component groups and bias conventions follow Liao et al. (TMLR 2026), Sec. 3 and Appendix A:
* revert C: start from the tuned weights, restore group C to its pretrained values.
* retain C: start from the tuned weights, restore every OTHER update (i.e. pretrained + delta of C only).
* VO revert restores V rows, V biases and the O weight, but leaves the output-projection bias tuned.
  VO retain keeps V, O weight AND the output-projection bias. So the two VO operators differ in one bias.

Parameter layout checked against open_clip 3.3.0 ViT-B-16 (torch nn.MultiheadAttention):
attn.in_proj_weight (3w, w) stacks Q, K, V rows in that order; attn.in_proj_bias (3w,) likewise;
attn.out_proj.{weight,bias}; mlp.c_fc / mlp.c_proj {weight,bias}; ln_1 / ln_2 {weight,bias}.
tests/test_components.py checks the Q, K, V order numerically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src import data as D
from src import model as M
from src.evaluate import compute_metrics, predict_logits
from src.finetune import build_eval_sets
from src.utils import get_device, load_config, run_metadata, save_json, set_seed

# Disjoint groups; together they cover every trained scalar exactly once.
GROUPS = ("qk", "v", "o_w", "o_b", "mlp", "ln1", "ln2")

# Named conditions: (mode, groups). Liao et al. Appendix A bias conventions.
CONDITIONS = {
    "revert_vo": ("revert", ("v", "o_w")),
    "revert_qk": ("revert", ("qk",)),
    "revert_mlp": ("revert", ("mlp",)),
    "revert_ln": ("revert", ("ln1", "ln2")),
    "retain_vo": ("retain", ("v", "o_w", "o_b")),
    "retain_qk": ("retain", ("qk",)),
    "retain_mlp": ("retain", ("mlp",)),
    "retain_ln": ("retain", ("ln1", "ln2")),
}


def param_masks(name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Boolean masks (same shape as tensor) saying which entries belong to which group.

    Only parameters inside a transformer block are expected; anything else raises, so a change
    in what is trained cannot slip through silently.
    """
    suffix = name.split(".resblocks.", 1)[-1].split(".", 1)[-1]  # e.g. "attn.in_proj_weight"

    def full(group):
        return {group: torch.ones_like(tensor, dtype=torch.bool)}

    if suffix in ("attn.in_proj_weight", "attn.in_proj_bias"):
        rows = tensor.shape[0]
        if rows % 3:
            raise ValueError(f"{name}: first dim {rows} is not 3 * width")
        w = rows // 3
        qk = torch.zeros_like(tensor, dtype=torch.bool)
        v = torch.zeros_like(tensor, dtype=torch.bool)
        qk[: 2 * w] = True
        v[2 * w:] = True
        return {"qk": qk, "v": v}
    if suffix == "attn.out_proj.weight":
        return full("o_w")
    if suffix == "attn.out_proj.bias":
        return full("o_b")
    if suffix.startswith("mlp."):
        return full("mlp")
    if suffix.startswith("ln_1."):
        return full("ln1")
    if suffix.startswith("ln_2."):
        return full("ln2")
    raise KeyError(f"No component group for parameter {name!r}")


def build_masks(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """param name -> group -> mask, checked to be a partition of every entry."""
    masks = {}
    for name, t in state.items():
        m = param_masks(name, t)
        cover = sum(mask.int() for mask in m.values())
        if not torch.all(cover == 1):
            raise AssertionError(f"{name}: groups do not partition the tensor")
        masks[name] = m
    return masks


def edit_state(pre: dict, ft: dict, masks: dict, mode: str, groups) -> dict[str, torch.Tensor]:
    """Weights for one condition.

    revert: tuned weights with `groups` set back to pretrained.
    retain: pretrained weights plus the update of `groups` only.
    """
    if mode not in ("revert", "retain"):
        raise ValueError(f"mode must be revert or retain, got {mode!r}")
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise KeyError(f"Unknown groups {unknown}")
    out = {}
    for name in ft:
        base, other = (ft[name], pre[name]) if mode == "revert" else (pre[name], ft[name])
        t = base.clone()
        for g in groups:
            m = masks[name].get(g)
            if m is not None:
                t[m] = other[m]
        out[name] = t
    return out


def condition_state(cond: str, pre: dict, ft: dict, masks: dict) -> dict:
    if cond == "pre":
        return {k: v.clone() for k, v in pre.items()}
    if cond == "ft":
        return {k: v.clone() for k, v in ft.items()}
    mode, groups = CONDITIONS[cond]
    return edit_state(pre, ft, masks, mode, groups)


def summarize_conditions(results: dict, target: str) -> dict:
    """Per condition: target accuracy, gain, held-out forgetting, and Liao-style ratios.

    gain(c)        = A_target(c) - A_target(pre)
    forgetting(c)  = A_heldout(pre) - A_heldout(c), per set and mean over held-out sets
    necessity      = A_target(ft) - A_target(revert C)                  (Liao Eq. 6, NecA)
    recovery       = gain(retain C) / gain(ft)                          (Liao Eq. 6, RecA)
    forgetting_share = forgetting(c) / forgetting(ft), mean over held-out sets.
      Unstable when forgetting(ft) is near zero; read it together with the raw numbers.
    """
    pre, ft = results["pre"], results["ft"]
    held = [k for k in ft if k != target]
    gain_ft = ft[target]["accuracy"] - pre[target]["accuracy"]
    forget_ft = {k: pre[k]["accuracy"] - ft[k]["accuracy"] for k in held}
    mean_forget_ft = sum(forget_ft.values()) / len(held) if held else 0.0
    out = {}
    for cond, res in results.items():
        gain = res[target]["accuracy"] - pre[target]["accuracy"]
        forget = {k: pre[k]["accuracy"] - res[k]["accuracy"] for k in held}
        mean_forget = sum(forget.values()) / len(held) if held else 0.0
        row = {"target_acc": res[target]["accuracy"], "gain": gain, "forgetting": forget,
               "mean_forgetting": mean_forget,
               "forgetting_share": (mean_forget / mean_forget_ft) if mean_forget_ft else None}
        if cond.startswith("revert_"):
            row["necessity"] = ft[target]["accuracy"] - res[target]["accuracy"]
        if cond.startswith("retain_"):
            row["recovery"] = gain / gain_ft if gain_ft else None
        out[cond] = row
    return out


def evaluate_sets(model, sets: dict, device, amp: bool) -> dict:
    """sets: name -> (loader, anchors). Anchors are fixed (text side frozen), so they are built once."""
    out = {}
    for name, (loader, anchors) in sets.items():
        logits, targets = predict_logits(model, loader, anchors, device, amp)
        out[name] = compute_metrics(logits, targets)
    return out


def result_path(cfg: dict, ckpt: str) -> Path:
    """Fixed name per checkpoint (no timestamp) so an interrupted run resumes where it stopped."""
    return Path(cfg["output"]["results_dir"]) / f"{cfg['experiment']}_{Path(ckpt).stem}.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    mcfg, dcfg = cfg["model"], cfg["data"]
    amp = cfg.get("amp", True)
    conditions = ["pre", "ft"] + list(cfg["conditions"])
    for c in cfg["conditions"]:
        if c not in CONDITIONS:
            raise KeyError(f"Unknown condition {c!r}. Known: {sorted(CONDITIONS)}")

    model, _, preprocess_val, tokenizer = M.load_clip(mcfg["name"], mcfg["pretrained"], device)
    M.freeze_for_finetune(model, cfg["train_last_k_blocks"])
    pre = M.trainable_state_dict(model)

    split_kw = dict(split_seed=dcfg.get("split_seed", 0), eurosat_test_fraction=dcfg.get("eurosat_test_fraction", 0.2),
                    split_mode=dcfg.get("split_mode", "permutation"))
    sets = {}
    for name, ds in build_eval_sets(dcfg, preprocess_val, split_kw).items():
        anchors = M.build_text_anchors(model, tokenizer, ds.classes, ds.templates, device=device)
        sets[name] = (D.make_loader(ds, dcfg["eval_batch_size"], dcfg["num_workers"], shuffle=False), anchors)

    for ckpt in cfg["checkpoints"]:
        ft = torch.load(ckpt, map_location="cpu")
        if set(ft) != set(pre):
            raise KeyError(f"{ckpt}: trained tensors differ from train_last_k_blocks={cfg['train_last_k_blocks']}")
        masks = build_masks(pre)
        path = result_path(cfg, ckpt)
        out = json.loads(path.read_text()) if (path.exists() and cfg.get("resume", True)) else {}
        out.update({"config": cfg, "meta": run_metadata(), "checkpoint": ckpt, "target": dcfg["target"]})
        out.setdefault("results", {})
        out["complete"] = False
        for cond in conditions:
            if cond in out["results"]:
                print(f"[{Path(ckpt).stem}] {cond}: already done, skipped")
                continue
            M.load_partial_state_dict(model, {k: v.to(device) for k, v in condition_state(cond, pre, ft, masks).items()})
            out["results"][cond] = evaluate_sets(model, sets, device, amp)
            accs = " ".join(f"{k}={v['accuracy']:.4f}" for k, v in out["results"][cond].items())
            print(f"[{Path(ckpt).stem}] {cond:11s} {accs}")
            save_json(out, path)
        out["summary"] = summarize_conditions(out["results"], dcfg["target"])
        out["complete"] = True
        save_json(out, path)
        print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
