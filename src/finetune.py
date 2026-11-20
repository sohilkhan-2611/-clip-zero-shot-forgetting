"""Fine-tune the last k visual blocks of OpenCLIP with frozen class-text anchors.

Run:  python -m src.finetune --config configs/finetune_flowers_vitb16.yaml [--set train.seed=1 ...]

Logits are logit_scale * cos(image feature, frozen text anchor of each class), trained with
cross-entropy. Only the chosen visual blocks change, so the text side (the zero-shot classifier
for every other dataset) is untouched. Saves only the trained tensors as the checkpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src import data as D
from src import model as M
from src.evaluate import _autocast, evaluate_dataset
from src.utils import get_device, load_config, results_path, run_metadata, save_json, set_seed


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup from 0 to 1, then cosine decay to 0."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_step(model, images, targets, anchors, optimizer, scaler, device, amp: bool, grad_clip: float | None):
    """One optimisation step. Returns (loss, number correct)."""
    model.train()
    with _autocast(device, amp):
        logits = M.image_logits(model, images, anchors)
        loss = F.cross_entropy(logits.float(), targets)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), (logits.argmax(1) == targets).sum().item()


def build_eval_sets(dcfg: dict, transform, split_kw: dict) -> dict:
    """Fixed evaluation sets, built once so base and fine-tuned models see the exact same images.

    Target: seed-0 uniform sample of data.target_eval_samples test images (2,000 in Liao et al.).
    Held-out: full test sets unless data.max_eval_samples is set (smoke runs).
    """
    target = dcfg["target"]
    sets = {}
    for name in [target] + [n for n in dcfg["eval_sets"] if n != target]:
        ds = D.build_dataset(name, "test", dcfg["root"], transform, **split_kw)
        n = dcfg.get("target_eval_samples") if name == target else dcfg.get("max_eval_samples")
        sets[name] = D.subsample(ds, n, dcfg.get("eval_sample_seed", 0))
    return sets


def summarize(base: dict, tuned: dict, target: str) -> dict:
    """Gain on the target and forgetting (base minus tuned accuracy) on each held-out set, in accuracy fractions."""
    out = {"target": target, "gain": tuned[target]["accuracy"] - base[target]["accuracy"], "forgetting": {}}
    for name in tuned:
        if name != target:
            out["forgetting"][name] = base[name]["accuracy"] - tuned[name]["accuracy"]
    if out["forgetting"]:
        out["mean_forgetting"] = sum(out["forgetting"].values()) / len(out["forgetting"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    seed = cfg["seed"]
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    amp = cfg.get("amp", True)

    model, preprocess_train, preprocess_val, tokenizer = M.load_clip(mcfg["name"], mcfg["pretrained"], device)
    trainable = M.freeze_for_finetune(model, tcfg["train_last_k_blocks"],
                                      tcfg.get("train_ln_post", False), tcfg.get("train_proj", False))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable tensors: {len(trainable)}, parameters: {n_train / 1e6:.2f}M")

    target = dcfg["target"]
    train_tf = preprocess_train if tcfg.get("augment", "open_clip") == "open_clip" else preprocess_val
    split_kw = dict(split_seed=dcfg.get("split_seed", 0), eurosat_test_fraction=dcfg.get("eurosat_test_fraction", 0.2),
                    split_mode=dcfg.get("split_mode", "permutation"))
    eval_sets = build_eval_sets(dcfg, preprocess_val, split_kw)

    def evaluate_all(tag):
        res = {}
        for name, ds in eval_sets.items():
            res[name] = evaluate_dataset(model, tokenizer, ds, device, dcfg["eval_batch_size"], dcfg["num_workers"], amp)
            print(f"[{tag}] {name:10s} acc={res[name]['accuracy']:.4f} n={res[name]['n']}")
        return res

    base = evaluate_all("base")  # same model before any update: the reference for gain and forgetting

    train_ds = D.build_dataset(target, "train", dcfg["root"], train_tf, **split_kw)
    train_ds = D.subsample(train_ds, dcfg.get("max_samples"), seed)
    print(f"train images: {len(train_ds)}")
    loader = D.make_loader(train_ds, tcfg["batch_size"], dcfg["num_workers"], shuffle=True, seed=seed, drop_last=True)
    anchors = M.build_text_anchors(model, tokenizer, train_ds.classes, train_ds.templates, device=device)

    if tcfg.get("optimizer", "adamw") != "adamw" or tcfg.get("schedule", "cosine") != "cosine":
        raise NotImplementedError("only optimizer: adamw and schedule: cosine are implemented")
    if tcfg.get("augment", "open_clip") not in ("open_clip", "none"):
        raise ValueError("train.augment must be open_clip or none")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, tcfg["warmup_steps"], total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    history = []
    for epoch in range(tcfg["epochs"]):
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), torch.as_tensor(targets).to(device)
            loss, correct = train_step(model, images, targets, anchors, optimizer, scaler, device, amp,
                                       tcfg.get("grad_clip"))
            scheduler.step()
            tot_loss += loss * len(targets)
            tot_correct += correct
            tot_n += len(targets)
        history.append({"epoch": epoch + 1, "train_loss": tot_loss / tot_n, "train_acc": tot_correct / tot_n,
                        "lr": scheduler.get_last_lr()[0]})
        print(history[-1])

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['experiment']}_seed{seed}.pt"
    torch.save(M.trainable_state_dict(model), ckpt_path)
    print(f"saved {ckpt_path}")

    tuned = evaluate_all("tuned")
    summary = summarize(base, tuned, target)
    print(f"gain on {target}: {summary['gain']:+.4f}; forgetting: "
          + ", ".join(f"{k} {v:+.4f}" for k, v in summary["forgetting"].items()))

    out = {"config": cfg, "meta": run_metadata(), "trainable_tensors": trainable,
           "trainable_params": n_train, "train_size": len(train_ds), "history": history,
           "checkpoint": str(ckpt_path), "target": target,
           "base": base, "results": tuned, "summary": summary}
    ref = cfg.get("liao_reference")
    if ref:
        out["liao_reference"] = ref
        print(f"Liao et al. reference for {target}: base {ref.get('base')}, full {ref.get('full')} "
              f"(ours: base {base[target]['accuracy']:.4f}, full {tuned[target]['accuracy']:.4f})")
    path = save_json(out, results_path(cfg, suffix=f"seed{seed}"))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
