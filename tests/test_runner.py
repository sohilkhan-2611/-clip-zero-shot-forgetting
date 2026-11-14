"""Sanity checks for the Colab run helpers (CPU, no downloads)."""

import json
import os
import time

import pytest
import torch

from src import model as M
from src import runner as R
from src.finetune import save_checkpoint


@pytest.fixture(scope="module")
def keys6():
    model, _, _, _ = M.load_clip("ViT-B-16-quickgelu", pretrained=None)
    names = set(M.freeze_for_finetune(model, 6))
    return names, M.trainable_state_dict(model)


def test_expected_keys_match_real_model(keys6):
    names, _ = keys6
    assert R.expected_trainable_keys(6) == names
    assert len(names) == 72


def test_checkpoint_status_and_should_train(tmp_path, keys6):
    names, state = keys6
    p = tmp_path / "ck_seed1.pt"
    assert R.checkpoint_status(p, names) == (False, "missing")
    assert R.should_train(p, names) is True

    save_checkpoint(state, p)
    assert not (tmp_path / "ck_seed1.pt.tmp").exists()     # temp file renamed away
    assert R.checkpoint_status(p, names)[0] is True
    assert R.should_train(p, names) is False                 # valid: skip, file untouched
    assert p.exists()

    p.write_bytes(p.read_bytes()[:1000])                     # truncated, like a reset mid-save
    ok, reason = R.checkpoint_status(p, names)
    assert not ok and reason.startswith("unreadable")
    assert R.should_train(p, names) is True
    assert not p.exists() and (tmp_path / "ck_seed1.pt.invalid").exists()   # kept, not overwritten


def test_checkpoint_wrong_keys_or_nan(tmp_path, keys6):
    names, state = keys6
    p = tmp_path / "a.pt"
    torch.save({k: v for i, (k, v) in enumerate(state.items()) if i > 0}, p)
    assert "wrong tensors" in R.checkpoint_status(p, names)[1]
    bad = dict(state)
    k0 = next(iter(bad))
    bad[k0] = bad[k0].clone().fill_(float("nan"))
    torch.save(bad, p)
    assert "non-finite" in R.checkpoint_status(p, names)[1]


def test_sync_json_copies_missing_and_newer(tmp_path):
    src, dst = tmp_path / "drive", tmp_path / "repo"
    src.mkdir()
    (src / "a.json").write_text('{"x": 1}')
    (src / "b.txt").write_text("not json")
    assert R.sync_json(src, dst) == ["a.json"]
    assert R.sync_json(src, dst) == []                       # nothing new
    time.sleep(0.01)
    (src / "a.json").write_text('{"x": 2}')
    os.utime(src / "a.json", (time.time() + 5, time.time() + 5))
    assert R.sync_json(src, dst) == ["a.json"]
    assert json.loads((dst / "a.json").read_text()) == {"x": 2}


def test_run_raises_on_failure(capsys):
    R.run("echo hello")
    assert "hello" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        R.run("exit 3")
