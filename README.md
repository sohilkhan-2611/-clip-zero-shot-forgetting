# CLIP zero-shot forgetting via interpretable weight directions

Research code for **"Predicting and Reducing Zero-Shot Forgetting in CLIP Fine-Tuning via Semantically Interpretable Weight Directions"** (M.Sc. project, TU Darmstadt).

Question: can semantic labels of fine-tuning delta directions, obtained without data, predict and reduce zero-shot forgetting in CLIP?

Status: **Phase 0 (setup)**. See `PROGRESS.md`.

## Layout

```
configs/      one YAML per experiment
src/
  utils.py      config loading with CLI overrides, seeds, JSON results with run metadata
  model.py      open_clip loading, frozen text anchors, logits, freezing, partial checkpoints
  data.py       datasets via clip_benchmark (same classnames and prompts as the published OpenCLIP table)
  evaluate.py   zero-shot accuracy, loss, correct-class margin; comparison with published values   (Phase 0)
  finetune.py   fine-tune the last k visual blocks with frozen class-text anchors                  (Phase 0)
notebooks/    Colab runners and plotting notebooks
results/      JSON outputs written by scripts, never edited by hand
tests/        CPU sanity checks (random weights, no downloads)
```

Later phases add `components.py`, `deltas.py`, `label.py`, `scores.py`, `intervene.py`, `predict.py`.

## Usage

```bash
pip install -e .[dev]
pytest -q                                                        # sanity checks, CPU
python -m src.evaluate --config configs/eval_zeroshot_vitb16.yaml
python -m src.finetune --config configs/finetune_flowers_vitb16.yaml --set seed=1
```

Any config value can be overridden with `--set key.sub=value`. Every run fixes its seed and writes a timestamped JSON (config, versions, git commit, metrics) to `results/`.
On Colab use `notebooks/00_phase0_colab_runner.ipynb`.

## Notes

* Fine-tuning checkpoints store only the trained tensors (the last k visual blocks), which is all later phases need to build weight deltas.
* Fine-tuning follows Liao et al. (TMLR 2026, Sec. 4 and App. A): OpenAI ViT-B/16 with QuickGELU (`ViT-B-16-quickgelu`), final six visual blocks trainable, final visual LayerNorm, projection, text tower and logit scale frozen, lr 1e-5, batch 48, 5 epochs, Flowers102 train + val, evaluation on a seed-0 sample of 2,000 test images.
* Settings Liao et al. do not state (optimizer, weight decay, warmup, schedule, augmentation) are marked `[ours]` in the configs.
* EuroSAT has no official split; like Liao et al. we use a seed-0 random permutation and an 80/20 division. The RNG is ours, so the exact images differ, and the comparison with the published zero-shot number is approximate.
* Each fine-tune evaluates the untouched model and the tuned model on the same fixed images and writes gain and forgetting to the JSON `summary`.

## Credits

This project builds directly on:

* **SITH**, Gentile et al., CVPR 2026: weight SVD and COMP concept labelling (the `sith` package is reused, not reimplemented, from Phase 2 on).
* **Liao et al.**, TMLR 2026: reversion and retention of CLIP fine-tuning deltas; our fine-tuning setup follows their recipe.
* **SAE-FT**, Morelli et al., 2026: evaluation protocol and sets used in Phase 5.
* [open_clip](https://github.com/mlfoundations/open_clip) for models and the published zero-shot table, and [clip_benchmark](https://github.com/LAION-AI/CLIP_benchmark) for dataset classnames and prompt templates.
