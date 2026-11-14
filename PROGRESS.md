# Progress

Source of truth for scope and done criteria: the phase plan doc. This file only tracks status.

## Phase 0: Setup (done 2026-11-14)

- [x] Repo structure, configs, seeds, JSON results
- [x] `data.py` (clip_benchmark loaders, Liao et al. splits)
- [x] `evaluate.py` (accuracy, loss, margin, comparison with published OpenCLIP table)
- [x] `finetune.py` (last 6 visual blocks, frozen text anchors, before/after evaluation)
- [x] CPU sanity tests and smoke test pass (20 tests, random weights, fake data)
- [x] Pretrained weights: OpenAI with QuickGELU (Liao et al. App. A)
- [x] Liao et al. and SAE-FT recipes noted; configs use Liao values, unstated ones marked `[ours]`
- [x] Colab: zero-shot eval of untouched ViT-B/16 roughly matches published values
- [x] Colab: one Flowers102 fine-tune runs end to end

### Results

Check 1, zero-shot of the untouched model (`results/phase0/eval_zeroshot_vitb16_20261114T004740Z.json`):

| Dataset | Ours | Published | Diff (pts) |
|---|---|---|---|
| CIFAR-10 | 0.9080 | 0.9077 | +0.03 |
| CIFAR-100 | 0.6695 | 0.6695 | 0.00 |
| DTD | 0.4479 | 0.4495 | -0.16 |
| Food101 | 0.8872 | 0.8872 | 0.00 |
| EuroSAT | 0.5430 | 0.5594 | -1.64 (own split) |
| Flowers102 | 0.7115 | 0.6907 | +2.08 (unexplained) |

Check 2, Flowers102 fine-tune, seed 0 (`results/phase0/finetune_flowers_vitb16_seed0_20261114T012159Z.json`):

| Dataset | Before | After | Change (pts) |
|---|---|---|---|
| Flowers102 (n = 2,000) | 0.6925 | 0.9710 | +27.9 |
| EuroSAT | 0.5430 | 0.4569 | -8.6 |
| Food101 | 0.8872 | 0.8766 | -1.1 |
| CIFAR-100 | 0.6695 | 0.6600 | -0.9 |
| CIFAR-10 | 0.9080 | 0.9051 | -0.3 |
| DTD | 0.4479 | 0.4489 | +0.1 |

Liao et al. Table 1 (B/16, Flowers102): 0.697 to 0.9735. Checkpoint `finetune_flowers_vitb16_seed0.pt` is in Google Drive.
Environment: Colab Tesla T4, Python 3.13, torch 2.14.1, open_clip 3.3.0.

### Carried into Phase 1
- Forgetting on CIFAR-10, CIFAR-100, Food101, DTD is 1 pt or less; EuroSAT carries most of it. Use several seeds.
- Open: harsher fine-tune as an extra setting? Flowers zero-shot +2.1 pts vs published? Keep EuroSAT for B/16? Augmentation?

## Phase 1: Pilot (current)

Question: which part of the trained blocks carries the gain, and which the forgetting?
Setup: Flowers102 target, 3 seeds, Liao recipe, full held-out sets; groups and bias rules from Liao et al. App. A.

- [x] `src/components.py`: revert / retain VO, QK, MLP, LayerNorm; resumable per-checkpoint JSON
- [x] `src/aggregate_phase1.py`: mean ± std over seeds, Markdown table
- [x] Tests (27 pass), including a numeric check that in_proj rows are Q, K, V
- [x] `notebooks/01_phase1_colab_runner.ipynb`
- [ ] Colab: fine-tune seeds 1 and 2
- [ ] Colab: components for seeds 0, 1, 2
- [ ] Table over seeds; decide VO-first or MLP-first (plan's decision rule)

**Done when:** one table of gain and forgetting per component with seed variation shown.

## Gate
Phases 2 to 6 start only after the professor checkpoint (after Phase 1).
