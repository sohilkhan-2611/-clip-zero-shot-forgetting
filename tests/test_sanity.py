"""Quick CPU sanity checks for every Phase 0 function. No downloads: the model uses random weights.

Run:  pytest -q
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import data as D
from src import model as M
from src.evaluate import compare_to_reference, compute_metrics
from src.finetune import cosine_with_warmup, summarize, train_step
from src.utils import apply_overrides, set_seed


@pytest.fixture(scope="module")
def clip():
    torch.manual_seed(0)
    model, _, _, tok = M.load_clip("ViT-B-16", pretrained=None)
    return model, tok


# utils ---------------------------------------------------------------------

def test_overrides_parse_types():
    cfg = apply_overrides({"data": {"max_samples": None}}, ["data.max_samples=256", "train.lr=1e-5", "x=null"])
    assert cfg["data"]["max_samples"] == 256
    assert cfg["train"]["lr"] == pytest.approx(1e-5)
    assert cfg["x"] is None


def test_set_seed_reproducible():
    set_seed(3); a = torch.randn(5)
    set_seed(3); b = torch.randn(5)
    assert torch.equal(a, b)


# metrics (known values) -------------------------------------------------------

def test_compute_metrics_known_values():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = torch.tensor([0, 2])
    m = compute_metrics(logits, targets)
    assert m["accuracy"] == pytest.approx(0.5)
    # margins: row0 2-1=1, row1 1-3=-2, mean -0.5
    assert m["margin"] == pytest.approx(-0.5)
    expected_loss = 0.5 * (-(2 - math.log(math.exp(2) + math.exp(1) + 1))
                           - (1 - math.log(1 + math.exp(3) + math.exp(1))))
    assert m["loss"] == pytest.approx(expected_loss, rel=1e-5)
    assert m["n"] == 2


def test_compare_to_reference():
    out = compare_to_reference({"cifar10": {"accuracy": 0.90}, "dtd": {"accuracy": 0.4}},
                               {"cifar10": 0.905}, tolerance_pp=1.0)
    assert out["cifar10"]["diff_pp"] == pytest.approx(-0.5)
    assert out["cifar10"]["within_tolerance"] is True
    assert out["dtd"]["published"] is None


# data ----------------------------------------------------------------------

def test_stratified_split_disjoint_and_balanced():
    targets = np.repeat(np.arange(4), 50)
    tr, te = D.stratified_split(targets, 0.2, seed=0)
    assert len(np.intersect1d(tr, te)) == 0
    assert len(tr) + len(te) == len(targets)
    assert all((targets[te] == c).sum() == 10 for c in range(4))
    tr2, te2 = D.stratified_split(targets, 0.2, seed=0)
    assert np.array_equal(te, te2)  # same seed, same split


def test_permutation_split_liao_protocol():
    tr, te = D.permutation_split(1000, 0.2, seed=0)
    assert len(tr) == 800 and len(te) == 200
    assert len(np.intersect1d(tr, te)) == 0 and len(np.union1d(tr, te)) == 1000
    tr2, te2 = D.permutation_split(1000, 0.2, seed=0)
    assert np.array_equal(te, te2)
    _, te3 = D.permutation_split(1000, 0.2, seed=1)
    assert not np.array_equal(te, te3)


def test_labeled_concat_keeps_metadata():
    class Part(torch.utils.data.Dataset):
        classes, templates = ["a", "b"], ["a photo of a {c}."]
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return torch.zeros(1), 0
    c = D.LabeledConcat([Part(3), Part(5)])
    assert len(c) == 8 and c.classes == ["a", "b"]
    assert len(D.subsample(c, 4, seed=0)) == 4


def test_subsample_keeps_metadata():
    class Toy(torch.utils.data.Dataset):
        classes, templates = ["a", "b"], ["a photo of a {c}."]
        def __len__(self): return 100
        def __getitem__(self, i): return torch.zeros(1), i % 2
    s = D.subsample(Toy(), 10, seed=0)
    assert len(s) == 10 and s.classes == ["a", "b"] and s.templates == Toy.templates


# model ---------------------------------------------------------------------

def test_structure_matches_assumptions(clip):
    model, _ = clip
    assert len(M.visual_blocks(model)) == 12
    assert model.visual.proj.shape == (768, 512)


def test_quickgelu_config_for_openai_weights():
    import open_clip
    assert open_clip.get_pretrained_cfg("ViT-B-16-quickgelu", "openai").get("quick_gelu") is True
    m = open_clip.create_model("ViT-B-16-quickgelu", pretrained=None)
    assert type(m.visual.transformer.resblocks[0].mlp.gelu).__name__ == "QuickGELU"


def test_freeze_keeps_block_layernorms_trainable(clip):
    # Liao et al.: final six blocks trainable, only the FINAL visual LayerNorm (ln_post) frozen
    model, _ = clip
    names = M.freeze_for_finetune(model, 6)
    assert "visual.transformer.resblocks.11.ln_1.weight" in names
    assert "visual.transformer.resblocks.11.ln_2.weight" in names
    assert not any(n.startswith("visual.ln_post") for n in names) and "visual.proj" not in names


def test_text_anchors_shape_and_norm(clip):
    model, tok = clip
    a = M.build_text_anchors(model, tok, ["cat", "dog", "car"], ["a photo of a {}.", "a {}."])
    assert a.shape == (512, 3)
    assert torch.allclose(a.norm(dim=0), torch.ones(3), atol=1e-5)
    # clip_benchmark style "{c}" templates must give the same anchors as "{}" ones
    b = M.build_text_anchors(model, tok, ["cat", "dog", "car"], ["a photo of a {c}.", "a {c}."])
    assert torch.allclose(a, b, atol=1e-6)


def test_image_logits_shape_and_scale(clip):
    model, tok = clip
    a = M.build_text_anchors(model, tok, ["cat", "dog"], ["a photo of a {}."])
    with torch.no_grad():
        logits = M.image_logits(model, torch.randn(2, 3, 224, 224), a)
    assert logits.shape == (2, 2)
    assert logits.abs().max() <= model.logit_scale.exp() + 1e-4  # cosine * scale


def test_freeze_last_k_blocks(clip):
    model, _ = clip
    names = M.freeze_for_finetune(model, 6)
    assert names and all(n.startswith("visual.transformer.resblocks.") for n in names)
    blocks = {int(n.split(".")[3]) for n in names}
    assert blocks == set(range(6, 12))
    expected = sum(p.numel() for b in list(M.visual_blocks(model))[6:] for p in b.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected
    assert not model.logit_scale.requires_grad and not model.token_embedding.weight.requires_grad


def test_train_step_changes_only_trainable(clip):
    model, tok = clip
    M.freeze_for_finetune(model, 1)
    frozen_before = model.visual.transformer.resblocks[0].mlp.c_fc.weight.clone()
    trained_before = model.visual.transformer.resblocks[11].mlp.c_fc.weight.clone()
    anchors = M.build_text_anchors(model, tok, ["cat", "dog"], ["a photo of a {}."])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss, _ = train_step(model, torch.randn(2, 3, 224, 224), torch.tensor([0, 1]), anchors, opt, scaler,
                         torch.device("cpu"), amp=False, grad_clip=1.0)
    assert math.isfinite(loss)
    assert torch.equal(model.visual.transformer.resblocks[0].mlp.c_fc.weight, frozen_before)
    assert not torch.equal(model.visual.transformer.resblocks[11].mlp.c_fc.weight, trained_before)


def test_checkpoint_roundtrip(clip):
    model, _ = clip
    M.freeze_for_finetune(model, 1)
    state = M.trainable_state_dict(model)
    assert state and all(k.startswith("visual.transformer.resblocks.11.") for k in state)
    with torch.no_grad():
        model.visual.transformer.resblocks[11].mlp.c_fc.weight.zero_()
    M.load_partial_state_dict(model, state)
    assert torch.equal(model.visual.transformer.resblocks[11].mlp.c_fc.weight,
                       state["visual.transformer.resblocks.11.mlp.c_fc.weight"])
    with pytest.raises(KeyError):
        M.load_partial_state_dict(model, {"not.a.key": torch.zeros(1)})


# schedule ------------------------------------------------------------------

def test_cosine_with_warmup():
    assert cosine_with_warmup(0, 10, 100) == pytest.approx(0.1)
    assert cosine_with_warmup(10, 10, 100) == pytest.approx(1.0)
    assert cosine_with_warmup(100, 10, 100) == pytest.approx(0.0, abs=1e-9)


def test_summarize_gain_and_forgetting():
    base = {"flowers": {"accuracy": 0.70}, "cifar10": {"accuracy": 0.90}, "dtd": {"accuracy": 0.45}}
    tuned = {"flowers": {"accuracy": 0.97}, "cifar10": {"accuracy": 0.85}, "dtd": {"accuracy": 0.46}}
    s = summarize(base, tuned, "flowers")
    assert s["gain"] == pytest.approx(0.27)
    assert s["forgetting"]["cifar10"] == pytest.approx(0.05)
    assert s["forgetting"]["dtd"] == pytest.approx(-0.01)  # negative = improved
    assert s["mean_forgetting"] == pytest.approx(0.02)
