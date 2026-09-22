"""local_jev -- a local, retraining-free decision engine built on SGLang.

Reproduces the Jev inference pattern: given a query and a fixed set of
allowed answers, return a probability distribution over those answers in a
single scoring request, instead of generating text.
"""

from .engine import DecideResult, GenerateResult, JevEngine
from .prompt import LabeledChoices, build_choices, build_prompt

__all__ = [
    "JevEngine",
    "DecideResult",
    "GenerateResult",
    "LabeledChoices",
    "build_choices",
    "build_prompt",
]
