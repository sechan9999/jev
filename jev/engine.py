"""JevEngine -- a local, retraining-free decision engine.

Two lanes run against the *same* SGLang server and the *same* model:

* ``decide()``            -- Jev-style fixed-answer scoring via ``/v1/score``.
                             One forward pass, read N next-token logits,
                             restricted softmax, return a distribution.
                             No tokens are generated.

* ``generate_response()`` -- ordinary autoregressive text generation via
                             ``/v1/chat/completions``.  The model writes an
                             answer token by token; we parse the choice out.

Comparing the two on identical inputs is the whole point of the benchmark.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

from .prompt import LabeledChoices, build_choices, build_prompt
from .tokenize_labels import resolve_label_token_ids


@dataclass
class DecideResult:
    choice: str                     # the winning answer (semantic, not a label)
    probabilities: dict[str, float]  # answer -> probability over the fixed set
    margin: float                   # top prob minus runner-up prob
    latency_ms: float


@dataclass
class GenerateResult:
    choice: str | None              # parsed answer, or None if nothing matched
    text: str                       # raw generated text
    latency_ms: float


class JevEngine:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:30000",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Cache label->token-id resolution per (sorted labels) so we do not
        # re-tokenize the alphabet on every single decision.
        self._token_id_cache: dict[tuple[str, ...], list[int]] = {}

    # ------------------------------------------------------------------
    # Jev-style scoring lane
    # ------------------------------------------------------------------
    def decide(
        self,
        query: str,
        answers,
        instruction: str | None = None,
    ) -> DecideResult:
        """Score a fixed set of answers and return a probability distribution.

        ``answers`` accepts the same forms as ``prompt.build_choices``:
        a list of answer strings, or a dict of ``answer -> meaning``.
        """
        choices = build_choices(answers)
        prompt = build_prompt(query, choices, instruction)
        label_token_ids = self._label_token_ids(choices)

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/score",
            json={
                "model": self.model,
                "query": prompt,
                # A single empty item scores the position right after the
                # query -- exactly the "Label:" continuation point.
                "items": [""],
                "label_token_ids": label_token_ids,
                "apply_softmax": True,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # scores[0] is aligned to label_token_ids, which is aligned to labels.
        scores = resp.json()["scores"][0]
        probabilities = {
            choices.answer_for(label): float(score)
            for label, score in zip(choices.labels, scores)
        }

        ranked = sorted(probabilities.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
        choice = max(probabilities, key=probabilities.get)

        return DecideResult(
            choice=choice,
            probabilities=probabilities,
            margin=margin,
            latency_ms=latency_ms,
        )

    def _label_token_ids(self, choices: LabeledChoices) -> list[int]:
        key = tuple(choices.labels)
        if key not in self._token_id_cache:
            self._token_id_cache[key] = resolve_label_token_ids(
                self.base_url, self.model, choices.labels, self.timeout
            )
        return self._token_id_cache[key]

    # ------------------------------------------------------------------
    # Standard generation lane (the baseline we compare against)
    # ------------------------------------------------------------------
    def generate_response(
        self,
        query: str,
        answers,
        instruction: str | None = None,
        max_tokens: int = 32,
    ) -> GenerateResult:
        """Ask the model to WRITE an answer, then parse a choice out of it.

        This is the ordinary LLM path: it generates tokens, so it is slower,
        and the answer has to be extracted from free text.
        """
        choices = build_choices(answers)
        prompt = build_prompt(query, choices, instruction)

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = resp.json()["choices"][0]["message"]["content"] or ""
        choice = self._parse_choice(text, choices)
        return GenerateResult(choice=choice, text=text, latency_ms=latency_ms)

    @staticmethod
    def _parse_choice(text: str, choices: LabeledChoices) -> str | None:
        """Recover a choice from generated text.

        The prompt asks the model to return only a label, so we accept a
        letter only when it is used *as a label* -- at the start of the reply,
        or immediately followed by a delimiter such as ``=`` or ``:``.  We do
        NOT match a bare letter mid-sentence, or the English article "a" in
        "a bug" would be read as label A.  If no label is found we fall back
        to any answer/meaning phrase in the first 100 characters, mirroring
        the article's lenient parser.
        """
        window = text[:100]
        upper = window.upper()
        labels = set(choices.labels)

        # 1) A label letter at the very start of the reply, e.g. "B" or
        #    "B = technical support" or "B) ...".
        m = re.match(r"\s*([A-Z])\b", upper)
        if m and m.group(1) in labels:
            return choices.answer_for(m.group(1))

        # 2) A label letter used as a label anywhere: followed by = : ) . -
        for m in re.finditer(r"\b([A-Z])\s*[=:).\-]", upper):
            if m.group(1) in labels:
                return choices.answer_for(m.group(1))

        # 3) Fall back to the answer/meaning text appearing verbatim.
        low = window.lower()
        for label in choices.labels:
            for phrase in (choices.answer_for(label), choices.label_to_meaning[label]):
                if phrase and phrase.lower() in low:
                    return choices.answer_for(label)
        return None
