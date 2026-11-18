from __future__ import annotations

from typing import Any, Dict, List

import torch


def _decode_tokens(tokenizer, ids) -> List[str]:
    out = []
    for t in ids:
        s = tokenizer.decode([int(t)], skip_special_tokens=False)
        out.append(s.replace("\n", "\\n"))
    return out


@torch.no_grad()
def lens_for_layer(model, hidden_states, tokenizer, layer: int, positions=None, topk: int = 5) -> Dict[str, Any]:
    """Project a layer's residual stream through final norm + lm_head.

    positions=None uses the last non-padding token; otherwise a list of token
    indices (e.g. image tokens) is averaged.
    """
    h = hidden_states[layer]
    if positions is None:
        h = h[0, -1]
    else:
        h = h[0, positions].mean(dim=0)
    dtype = next(model.parameters()).dtype
    norm = model.model.language_model.norm
    logits = model.lm_head(norm(h.to(dtype))).float()
    probs = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(probs, k=topk)
    return {
        "layer": int(layer),
        "tokens": _decode_tokens(tokenizer, ids.tolist()),
        "probs": [round(float(v), 5) for v in vals.tolist()],
        "top_id": int(ids[0]),
        "top_prob": round(float(vals[0]), 5),
    }


@torch.no_grad()
def lens_table(model, hidden_states, tokenizer, layers: List[int], topk: int = 4) -> List[Dict[str, Any]]:
    return [lens_for_layer(model, hidden_states, tokenizer, int(l), topk=topk) for l in layers]


@torch.no_grad()
def lens_image_tokens(model, hidden_states, tokenizer, layer: int, image_positions, topk: int = 4):
    return lens_for_layer(model, hidden_states, tokenizer, layer, positions=image_positions, topk=topk)
