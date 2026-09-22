"""Benchmark: Jev-style scoring vs ordinary text generation.

Both lanes hit the SAME SGLang server and the SAME model, so the only
variable is the inference pattern:

  * jev lane -- decide()            : one forward pass, no tokens generated
  * llm lane -- generate_response() : autoregressive, then parse the choice

The two lanes are released together by a barrier and then run concurrently,
each firing its next request as soon as the previous one returns.  SGLang
schedules the mixed load through continuous batching on one GPU.

Run the server first (see run_server.sh), then:

    python benchmark.py --n 100
"""

from __future__ import annotations

import argparse
import os
import statistics
import threading
from queue import Queue

from datasets import all_cases
from jev import JevEngine

BASE_URL = os.environ.get("JEV_BASE_URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("JEV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def run_jev(engine: JevEngine, case: dict) -> dict:
    r = engine.decide(case["query"], case["answers"])
    return {
        "lane": "jev",
        "latency_ms": r.latency_ms,
        "correct": r.choice == case["label"],
    }


def run_llm(engine: JevEngine, case: dict) -> dict:
    r = engine.generate_response(case["query"], case["answers"], max_tokens=32)
    return {
        "lane": "llm",
        "latency_ms": r.latency_ms,
        # An unparseable answer counts as wrong -- that failure mode is part
        # of what the generation lane costs you.
        "correct": r.choice == case["label"],
    }


def summarize(results: list[dict], lane: str) -> dict:
    rows = [r for r in results if r["lane"] == lane]
    lat = sorted(r["latency_ms"] for r in rows)
    n = len(rows)
    return {
        "lane": lane,
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "mean_ms": statistics.mean(lat) if lat else 0.0,
        "median_ms": statistics.median(lat) if lat else 0.0,
        "p95_ms": lat[int(0.95 * (n - 1))] if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="number of cases per lane")
    args = parser.parse_args()

    engine = JevEngine(model=MODEL, base_url=BASE_URL)
    cases = all_cases(args.n)

    updates: Queue = Queue()
    starting_line = threading.Barrier(2)

    def worker(runner) -> None:
        starting_line.wait()  # release both lanes at the same instant
        for case in cases:
            updates.put(runner(engine, case))

    workers = [
        threading.Thread(target=worker, args=(run_jev,), daemon=True),
        threading.Thread(target=worker, args=(run_llm,), daemon=True),
    ]
    for t in workers:
        t.start()
    for t in workers:
        t.join()

    results = [updates.get() for _ in range(updates.qsize())]

    jev, llm = summarize(results, "jev"), summarize(results, "llm")
    speedup = llm["median_ms"] / jev["median_ms"] if jev["median_ms"] else float("nan")

    header = f"{'lane':<6}{'n':>5}{'accuracy':>11}{'median_ms':>12}{'mean_ms':>11}{'p95_ms':>11}"
    print(f"\nModel: {MODEL}   cases/lane: {args.n}\n")
    print(header)
    print("-" * len(header))
    for s in (jev, llm):
        print(
            f"{s['lane']:<6}{s['n']:>5}{s['accuracy']:>11.2%}"
            f"{s['median_ms']:>12.1f}{s['mean_ms']:>11.1f}{s['p95_ms']:>11.1f}"
        )
    print(f"\nScoring is ~{speedup:.1f}x faster than generation at the median.")


if __name__ == "__main__":
    main()
