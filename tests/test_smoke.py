"""End-to-end smoke test of src.evaluate and src.finetune on fake images (random weights, CPU, no downloads).

Checks that configs parse, the scripts run, and JSON plus a checkpoint land on disk.
It says nothing about accuracy; real numbers only come from Colab runs on real data.
"""

import json

import torch

from src import data as D
from src import evaluate, finetune


class FakeDS(torch.utils.data.Dataset):
    classes = ["cat", "dog", "car"]
    templates = ["a photo of a {c}."]

    def __init__(self, transform=None):
        self.n = 8

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.randn(3, 224, 224), i % 3


def _fake_build(name, split, root, transform, **kw):
    return FakeDS(transform)


def test_evaluate_keeps_finished_sets_when_interrupted(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky_build(name, split, root, transform, **kw):
        calls["n"] += 1
        if name == "food101":
            raise KeyboardInterrupt  # like stopping a slow download
        return FakeDS(transform)

    monkeypatch.setattr(D, "build_dataset", flaky_build)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        evaluate.main(["--config", "configs/eval_zeroshot_vitb16.yaml", "--set",
                       f"output.results_dir={tmp_path}", "model.pretrained=null", "device=cpu",
                       "data.num_workers=0", "amp=false", "data.batch_size=4", "reference=null",
                       "data.eval_sets=[cifar10,dtd,food101]"])
    (f,) = list(tmp_path.glob("*.json"))
    saved = json.loads(f.read_text())
    assert set(saved["results"]) == {"cifar10", "dtd"} and saved["complete"] is False


def test_evaluate_and_finetune_run(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "build_dataset", _fake_build)
    common = [f"output.results_dir={tmp_path}/results", "model.pretrained=null", "device=cpu",
              "data.num_workers=0", "amp=false"]

    out = evaluate.main(["--config", "configs/eval_zeroshot_vitb16.yaml", "--set", *common,
                         "data.eval_sets=[cifar10,dtd]", "data.batch_size=4", "reference=null"])
    assert set(out["results"]) == {"cifar10", "dtd"} and out["results"]["cifar10"]["n"] == 8
    assert out["complete"] is True

    out = finetune.main(["--config", "configs/finetune_flowers_vitb16.yaml", "--set", *common,
                         f"output.checkpoint_dir={tmp_path}/ckpt", "train.epochs=1", "train.batch_size=4",
                         "train.warmup_steps=1", "data.eval_sets=[cifar10]", "data.eval_batch_size=4"])
    assert len(out["history"]) == 1 and set(out["results"]) == {"flowers", "cifar10"}
    assert set(out["base"]) == {"flowers", "cifar10"} and "cifar10" in out["summary"]["forgetting"]
    assert out["results"]["flowers"]["n"] == 8  # target_eval_samples=2000 > 8 keeps all
    state = torch.load(out["checkpoint"])
    assert all(k.startswith("visual.transformer.resblocks.") for k in state)

    files = list((tmp_path / "results").glob("*.json"))
    assert len(files) == 2
    for f in files:
        json.loads(f.read_text())  # valid JSON
