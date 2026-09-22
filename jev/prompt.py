"""Prompt construction for fixed-answer scoring.

The whole trick of Jev-style scoring is that every allowed answer is
represented by a *single-token* letter label (A, B, C, ...) placed at the
very end of the prompt.  We put the human-readable meaning of each label
inside the prompt so the model still "sees" the semantics, but the only
thing we score afterwards is the one token that would come next.

Nothing in this module talks to the network -- it is pure text.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field


# Single-character labels, in a fixed order.  We never use more labels than
# the caller has choices, and 26 is already far more categories than a
# routing decision should ever have.
LABEL_ALPHABET = list(string.ascii_uppercase)


@dataclass
class LabeledChoices:
    """A bidirectional mapping between letter labels and semantic answers.

    - ``label_to_meaning``  : "A" -> "billing and payments"   (shown to model)
    - ``label_to_answer``   : "A" -> "billing"                (returned to app)

    The application only ever deals in ``answer`` strings.  Labels are an
    internal detail of the scoring client, exactly as the article describes.
    """

    label_to_meaning: dict[str, str] = field(default_factory=dict)
    label_to_answer: dict[str, str] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        # Insertion order is preserved (Python 3.7+), which keeps labels,
        # tokenization, scores, and results all aligned to the same order.
        return list(self.label_to_meaning.keys())

    def answer_for(self, label: str) -> str:
        return self.label_to_answer[label]


def build_choices(answers) -> LabeledChoices:
    """Assign a letter label to each answer.

    ``answers`` may be either:

    * a list of strings -- the answer is used as its own meaning, e.g.
      ``["billing", "technical support", "account access"]``
    * a dict ``answer -> meaning`` -- the meaning (a longer description) is
      what the model reads, while the answer is what the app gets back, e.g.
      ``{"billing": "billing questions and payment problems", ...}``
    """
    if isinstance(answers, dict):
        items = list(answers.items())
    else:
        items = [(a, a) for a in answers]

    if len(items) > len(LABEL_ALPHABET):
        raise ValueError(
            f"{len(items)} answers exceeds {len(LABEL_ALPHABET)} single-letter labels"
        )

    choices = LabeledChoices()
    for label, (answer, meaning) in zip(LABEL_ALPHABET, items):
        choices.label_to_meaning[label] = meaning
        choices.label_to_answer[label] = answer
    return choices


def build_prompt(query: str, choices: LabeledChoices, instruction: str | None = None) -> str:
    """Render the scoring prompt.

    The prompt *must* end at ``Label:`` so that the next-token position is
    exactly where the model would emit A, B, or C.  That is the single
    position whose logits we read.
    """
    instruction = instruction or "Which label best matches the input?"
    choice_lines = "\n".join(
        f"{label} = {meaning}" for label, meaning in choices.label_to_meaning.items()
    )
    return (
        f"Input:\n{query}\n\n"
        f"Question:\n{instruction}\n\n"
        f"Allowed labels:\n{choice_lines}\n\n"
        f"Return only the label.\n"
        f"Label:"
    )
