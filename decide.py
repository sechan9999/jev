"""Single-request walkthrough of fixed-answer scoring.

Run the SGLang server first (see run_server.sh), then:

    python decide.py

This mirrors the five-step flow:
  1. define the choices and build the prompt
  2. resolve the label token IDs (and verify each is a single token)
  3. send one /v1/score request
  4. read the restricted-softmax distribution
  5. map labels back to semantic answers
"""

from __future__ import annotations

import json
import os

from jev import JevEngine

BASE_URL = os.environ.get("JEV_BASE_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("JEV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def main() -> None:
    engine = JevEngine(model=MODEL, base_url=BASE_URL)

    # Step 1 -- the application already knows the valid answers.
    # Keys are what the app gets back; values are what the model reads.
    answers = {
        "billing and payments": "billing questions and payment problems",
        "technical support": "product errors and technical failures",
        "account access": "login, password, and account access problems",
    }
    ticket = "I was charged twice for the same subscription."

    # Steps 2-5 all happen inside decide(): tokenize + validate labels,
    # one /v1/score call, restricted softmax, map back to answers.
    result = engine.decide(
        query=ticket,
        answers=answers,
        instruction="Which category matches the ticket?",
    )

    print(
        json.dumps(
            {
                "decision": result.choice,
                "margin": round(result.margin, 4),
                "latency_ms": round(result.latency_ms, 1),
                "probabilities": {k: round(v, 4) for k, v in result.probabilities.items()},
            },
            indent=2,
        )
    )

    # The routing threshold lives in application code, not in the model.
    THRESHOLD, MIN_MARGIN = 0.70, 0.20
    top = result.probabilities[result.choice]
    if top >= THRESHOLD and result.margin >= MIN_MARGIN:
        print(f"\n=> auto-route to {result.choice!r} (confident)")
    else:
        print(f"\n=> send to human review (top={top:.2f}, margin={result.margin:.2f})")


if __name__ == "__main__":
    main()
