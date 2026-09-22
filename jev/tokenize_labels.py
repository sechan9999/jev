"""Resolve letter labels to vocabulary token IDs, and verify each is a
single token.

Fixed-answer scoring reads exactly ONE logit per answer, at ONE output
position.  That only works if every label is a single token in the model's
vocabulary.  A visible character is not guaranteed to be one token (leading
spaces, byte-pair merges, and chat-template control tokens all interfere),
so we ask the server's own tokenizer and reject anything that splits.
"""

from __future__ import annotations

import requests


def resolve_label_token_ids(
    base_url: str,
    model: str,
    labels: list[str],
    timeout: int = 30,
) -> list[int]:
    """Return one token ID per label, in the same order as ``labels``.

    Raises ``ValueError`` if any label does not encode to exactly one token,
    so a bad label surfaces at setup time rather than as silently wrong
    scores at inference time.
    """
    token_ids: list[int] = []
    for label in labels:
        resp = requests.post(
            f"{base_url}/tokenize",
            json={
                "model": model,
                "prompt": label,
                # We are checking the label in isolation, so no BOS/EOS.
                "add_special_tokens": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        ids = resp.json()["tokens"]
        if len(ids) != 1:
            raise ValueError(
                f"label {label!r} is not a single token for model {model!r}: {ids}. "
                "Pick a different label character."
            )
        token_ids.append(ids[0])
    return token_ids
