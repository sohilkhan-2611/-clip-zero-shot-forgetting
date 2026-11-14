"""Sanity checks for Phase 1 component editing (CPU, random weights)."""

import json

import pytest
import torch
import torch.nn.functional as F

from src import components as C
from src import model as M


@pytest.fixture(scope="module")
def states():
    torch.manual_seed(0)
    model, _, _, _ = M.load_clip("ViT-B-16-quickgelu", pretrained=None)
    M.freeze_for_finetune(model, 2)
    pre = M.trainable_state_dict(model)
    ft = {k: v + 0.01 * torch.randn_like(v) for k, v in pre.items()}
    return model, pre, ft


def test_masks_partition_every_entry(states):
    _, pre, _ = states
    masks = C.build_masks(pre)  # raises if any entry is in 0 or 2 groups
    assert set(masks) == set(pre)
    w = 768
    m = masks["visual.transformer.resblocks.11.attn.in_proj_weight"]
    assert m["qk"][: 2 * w].all() and not m["qk"][2 * w:].any()
    assert m["v"][2 * w:].all() and not m["v"][: 2 * w].any()
    assert masks["visual.transformer.resblocks.11.attn.out_proj.bias"].keys() == {"o_b"}
    assert masks["visual.transformer.resblocks.11.ln_2.weight"].keys() == {"ln2"}


def test_unknown_parameter_raises():
    with pytest.raises(KeyError):
        C.param_masks("visual.transformer.resblocks.0.something.weight", torch.zeros(3))


def test_in_proj_order_is_q_k_v():
    # If Q and K rows are zero, attention is uniform, so each output token is
    # out_proj(mean over tokens of V x). This only holds if rows 2w:3w really are V.
    torch.manual_seed(0)
    w, n = 8, 5
    attn = torch.nn.MultiheadAttention(w, num_heads=2, batch_first=True)
    with torch.no_grad():
        attn.in_proj_weight[: 2 * w] = 0
        attn.in_proj_bias[: 2 * w] = 0
    x = torch.randn(1, n, w)
    out, _ = attn(x, x, x, need_weights=False)
    v = F.linear(x, attn.in_proj_weight[2 * w:], attn.in_proj_bias[2 * w:])
    expected = attn.out_proj(v.mean(dim=1, keepdim=True)).expand(1, n, w)
    assert torch.allclose(out, expected, atol=1e-5)


def test_revert_and_retain_identities(states):
    _, pre, ft = states
    masks = C.build_masks(pre)
    all_groups = C.GROUPS
    rev_all = C.edit_state(pre, ft, masks, "revert", all_groups)
    ret_all = C.edit_state(pre, ft, masks, "retain", all_groups)
    assert all(torch.equal(rev_all[k], pre[k]) for k in pre)
    assert all(torch.equal(ret_all[k], ft[k]) for k in pre)
    # revert(C) + retain(C) = pre + ft, entry by entry, for any C
    for cond in ("revert_mlp", "revert_qk", "revert_ln"):
        groups = C.CONDITIONS[cond][1]
        r = C.edit_state(pre, ft, masks, "revert", groups)
        k = C.edit_state(pre, ft, masks, "retain", groups)
        assert all(torch.allclose(r[n] + k[n], pre[n] + ft[n]) for n in pre)


def test_liao_vo_bias_convention(states):
    _, pre, ft = states
    masks = C.build_masks(pre)
    ob = "visual.transformer.resblocks.11.attn.out_proj.bias"
    ow = "visual.transformer.resblocks.11.attn.out_proj.weight"
    ipw = "visual.transformer.resblocks.11.attn.in_proj_weight"
    rev = C.condition_state("revert_vo", pre, ft, masks)
    assert torch.equal(rev[ob], ft[ob])           # output bias stays tuned
    assert torch.equal(rev[ow], pre[ow])
    assert torch.equal(rev[ipw][1536:], pre[ipw][1536:]) and torch.equal(rev[ipw][:1536], ft[ipw][:1536])
    ret = C.condition_state("retain_vo", pre, ft, masks)
    assert torch.equal(ret[ob], ft[ob])           # retention keeps the output bias update
    assert torch.equal(ret[ipw][:1536], pre[ipw][:1536])
    mlp = "visual.transformer.resblocks.11.mlp.c_fc.weight"
    assert torch.equal(ret[mlp], pre[mlp]) and torch.equal(rev[mlp], ft[mlp])


def test_summarize_conditions_known_values():
    def r(t, a, b):
        return {"flowers": {"accuracy": t}, "cifar10": {"accuracy": a}, "eurosat": {"accuracy": b}}
    res = {"pre": r(0.70, 0.90, 0.54), "ft": r(0.97, 0.89, 0.46),
           "revert_mlp": r(0.85, 0.90, 0.50), "retain_mlp": r(0.92, 0.895, 0.48)}
    s = C.summarize_conditions(res, "flowers")
    assert s["ft"]["gain"] == pytest.approx(0.27)
    assert s["ft"]["mean_forgetting"] == pytest.approx((0.01 + 0.08) / 2)
    assert s["revert_mlp"]["necessity"] == pytest.approx(0.12)
    assert s["retain_mlp"]["recovery"] == pytest.approx(0.22 / 0.27)
    assert s["revert_mlp"]["forgetting_share"] == pytest.approx(((0.0 + 0.04) / 2) / 0.045)


def test_components_and_aggregate_run(tmp_path, monkeypatch, states):
    """End to end on fake images: two fake checkpoints, resume, then the seed table."""
    from src import aggregate_phase1, data as D
    from tests.test_smoke import FakeDS

    monkeypatch.setattr(D, "build_dataset", lambda name, split, root, transform, **kw: FakeDS(transform))
    model, pre, _ = states
    ckpts = []
    for s in range(2):
        g = torch.Generator().manual_seed(s)
        p = tmp_path / f"finetune_flowers_vitb16_seed{s}.pt"
        torch.save({k: v + 0.01 * torch.randn(v.shape, generator=g) for k, v in pre.items()}, p)
        ckpts.append(str(p))
    args = ["--config", "configs/phase1_components_flowers.yaml", "--set", "model.pretrained=null", "device=cpu",
            "amp=false", "train_last_k_blocks=2", f"checkpoints=[{','.join(ckpts)}]", "data.num_workers=0",
            "data.eval_batch_size=4", "data.eval_sets=[cifar10]", f"output.results_dir={tmp_path}/res",
            "conditions=[revert_mlp,retain_vo]"]
    C.main(args)
    files = sorted((tmp_path / "res").glob("*.json"))
    assert len(files) == 2
    d = json.loads(files[0].read_text())
    assert d["complete"] and set(d["results"]) == {"pre", "ft", "revert_mlp", "retain_vo"}
    assert "recovery" in d["summary"]["retain_vo"] and "necessity" in d["summary"]["revert_mlp"]
    # resume: a second run skips everything already done and leaves the numbers unchanged
    before = d["results"]
    C.main(args)
    assert json.loads(files[0].read_text())["results"] == before
    out = tmp_path / "summary.json"
    aggregate_phase1.main(["--inputs", str(tmp_path / "res" / "*.json"), "--out", str(out)])
    agg = json.loads(out.read_text())
    assert agg["n_seeds"] == 2 and agg["aggregate"]["ft"]["gain"]["n"] == 2


@pytest.mark.parametrize("path", ["configs/phase1_components_flowers.yaml", "configs/phase1_components_eurosat.yaml"])
def test_phase1_configs(path):
    from src.utils import load_config
    cfg = load_config(path)
    d = cfg["data"]
    assert d["target"] not in d["eval_sets"]          # a target is never also a held-out set
    assert len(cfg["checkpoints"]) == 3 and all(d["target"] in c for c in cfg["checkpoints"])
    assert set(cfg["conditions"]) <= set(C.CONDITIONS)


def test_components_resume_only_runs_missing_conditions(tmp_path, monkeypatch, states):
    """A file with some finished conditions (e.g. from an interrupted run) is continued, not redone."""
    from src import data as D
    from tests.test_smoke import FakeDS

    monkeypatch.setattr(D, "build_dataset", lambda name, split, root, transform, **kw: FakeDS(transform))
    _, pre, _ = states
    ck = tmp_path / "finetune_flowers_vitb16_seed1.pt"
    torch.save({k: v + 0.01 for k, v in pre.items()}, ck)
    base = ["--config", "configs/phase1_components_flowers.yaml", "--set", "model.pretrained=null", "device=cpu",
            "amp=false", "train_last_k_blocks=2", f"checkpoints=[{ck}]", "data.num_workers=0",
            "data.eval_batch_size=4", "data.eval_sets=[cifar10]", f"output.results_dir={tmp_path}/res"]
    C.main(base + ["conditions=[revert_mlp]"])           # first session: pre, ft, revert_mlp
    (f,) = list((tmp_path / "res").glob("*.json"))
    first = json.loads(f.read_text())["results"]

    calls = []
    real = C.evaluate_sets
    monkeypatch.setattr(C, "evaluate_sets", lambda *a, **k: calls.append(1) or real(*a, **k))
    C.main(base + ["conditions=[revert_mlp,retain_mlp]"])  # second session adds one condition
    assert len(calls) == 1                                   # only retain_mlp was evaluated
    d = json.loads(f.read_text())
    assert d["complete"] and set(d["results"]) == {"pre", "ft", "revert_mlp", "retain_mlp"}
    assert all(d["results"][c] == first[c] for c in first)  # finished conditions untouched
