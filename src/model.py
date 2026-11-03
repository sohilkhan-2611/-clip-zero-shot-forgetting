"""Model loading, frozen text anchors, logits, and freezing for the open_clip ViT.

Checked against open_clip 3.3.0: for ViT-B-16 the image tower is
model.visual (VisionTransformer) with 12 blocks in model.visual.transformer.resblocks,
then model.visual.ln_post and the parameter model.visual.proj.
"""

from __future__ import annotations

import open_clip
import torch
import torch.nn.functional as F


def load_clip(name: str, pretrained: str | None, device: torch.device | str = "cpu"):
    """Return (model, preprocess_train, preprocess_val, tokenizer). pretrained=None gives random weights (tests)."""
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(name)
    model.eval()
    return model, preprocess_train, preprocess_val, tokenizer


@torch.no_grad()
def build_text_anchors(model, tokenizer, classnames, templates, device="cpu") -> torch.Tensor:
    """Class text embeddings, averaged over templates and L2 normalised. Shape (embed_dim, num_classes).

    Thin wrapper over open_clip.build_zero_shot_classifier so we reuse its exact prompt averaging.
    open_clip calls template.format(c) (positional) but clip_benchmark templates use "{c}",
    so string templates are turned into callables that accept either style.
    """
    fns = [_template_fn(t) for t in templates]
    anchors = open_clip.build_zero_shot_classifier(
        model, tokenizer=tokenizer, classnames=list(classnames), templates=fns,
        num_classes_per_batch=10, device=device, use_tqdm=False,
    )
    return anchors.detach()


def _template_fn(t):
    if callable(t):
        return t
    if "{c}" in t:
        return lambda c, t=t: t.format(c=c)
    return lambda c, t=t: t.format(c)


def image_logits(model, images: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """logit_scale * cos(image, class anchor). Shape (batch, num_classes).

    Uses the model's own logit_scale (frozen), so logits match zero-shot inference.
    """
    feats = model.encode_image(images)
    feats = F.normalize(feats, dim=-1)
    return model.logit_scale.exp() * feats @ anchors.to(feats.dtype)


def visual_blocks(model):
    return model.visual.transformer.resblocks


def freeze_for_finetune(
    model, train_last_k_blocks: int, train_ln_post: bool = False, train_proj: bool = False
) -> list[str]:
    """Freeze everything, then unfreeze the last k visual blocks (and optionally ln_post / proj).

    Returns the names of trainable parameters. Text tower and logit_scale stay frozen,
    so the class text anchors are fixed, as in the plan (Liao et al. setting).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = visual_blocks(model)
    n = len(blocks)
    if not 0 < train_last_k_blocks <= n:
        raise ValueError(f"train_last_k_blocks must be in 1..{n}, got {train_last_k_blocks}")
    for block in list(blocks)[n - train_last_k_blocks:]:
        for p in block.parameters():
            p.requires_grad_(True)
    if train_ln_post:
        for p in model.visual.ln_post.parameters():
            p.requires_grad_(True)
    if train_proj and model.visual.proj is not None:
        model.visual.proj.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    """Only the trainable tensors, on CPU. This is what we checkpoint (deltas are built from it later)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_partial_state_dict(model, state: dict[str, torch.Tensor]) -> list[str]:
    """Load a checkpoint that holds only some tensors. Fails loudly on any unknown key."""
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise KeyError(f"Checkpoint has keys the model does not: {result.unexpected_keys[:5]}")
    return list(state.keys())
