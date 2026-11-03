# Progress

Source of truth for scope and done criteria: the phase plan doc. This file only tracks status.

## Phase 0: Setup (current)

- [x] Repo structure, configs, seeds, JSON results
- [x] `data.py` (clip_benchmark loaders, own EuroSAT split)
- [x] `evaluate.py` (accuracy, loss, margin, comparison with published OpenCLIP table)
- [x] `finetune.py` (last 6 visual blocks, frozen text anchors)
- [x] CPU sanity tests and smoke test pass (random weights, fake data)
- [ ] Decide pretrained weights (`openai` vs `laion2b_s34b_b88k`) to match Liao et al.
- [ ] Note Liao et al. and SAE-FT recipes; replace placeholder hyperparameters in configs
- [ ] Colab: zero-shot eval of untouched ViT-B/16 roughly matches published values
- [ ] Colab: one Flowers102 fine-tune runs end to end

**Done when:** the last two boxes are ticked, with result JSONs in `results/phase0/`.

## Gate
Phases 2 to 6 start only after the professor checkpoint (after Phase 1).
