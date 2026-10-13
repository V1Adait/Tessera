"""Score the question classifiers on eval/router/questions.jsonl and print a comparison.

    .venv/Scripts/python eval/router/run_eval.py [--repeats 2] [--threshold 0.8]

Scoring
  strict   the label equals the gold label
  lenient  the label is the gold label OR one listed in `also_ok`
Calibration (does confidence tell us when to hand off?)
  A prediction below --threshold is "flagged" for the LLM handoff. A good classifier flags its
  wrong answers and leaves its right ones alone.
Every number printed here comes from this run; results are also saved under eval/router/results/.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from classifiers import (
    CLASSES,
    Classifier,
    JevClassifier,
    LLMClassifier,
    Prediction,
)

from backend.app.config import get_settings


def load_questions() -> list[dict]:
    with open(HERE / "questions.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert all(r["label"] in CLASSES for r in rows), "gold label outside CLASSES"
    return rows


def run(
    classifier: Classifier, rows: list[dict], repeats: int
) -> list[list[Prediction]]:
    """predictions[repeat][question]"""
    return [[classifier.classify(r["question"]) for r in rows] for _ in range(repeats)]


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"


def report(
    name: str, rows: list[dict], runs: list[list[Prediction]], threshold: float
) -> dict:
    first = runs[0]
    errors = [(r["id"], p.error) for r, p in zip(rows, first, strict=True) if p.error]
    strict = [p.label == r["label"] for r, p in zip(rows, first, strict=True)]
    lenient = [
        p.label is not None and (p.label == r["label"] or p.label in r["also_ok"])
        for r, p in zip(rows, first, strict=True)
    ]
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    if errors:
        print(f"ERRORS ({len(errors)}): {errors[:3]}")

    print(f"strict accuracy   {pct(sum(strict), len(rows))}")
    print(f"lenient accuracy  {pct(sum(lenient), len(rows))}")
    for level in ("clear", "hard"):
        idx = [i for i, r in enumerate(rows) if r["difficulty"] == level]
        print(
            f"  {level:6s} strict {pct(sum(strict[i] for i in idx), len(idx)):12s}"
            f" lenient {pct(sum(lenient[i] for i in idx), len(idx))}"
        )

    decomp = [
        p.needs_decomposition == r["needs_decomposition"]
        for r, p in zip(rows, first, strict=True)
        if p.needs_decomposition is not None
    ]
    print(f"needs_decomposition correct  {pct(sum(decomp), len(decomp))}")

    # calibration: confidence on right vs wrong, and what a handoff threshold would do
    conf_ok = [
        p.confidence
        for p, ok in zip(first, strict, strict=True)
        if ok and p.confidence is not None
    ]
    conf_bad = [
        p.confidence
        for p, ok in zip(first, strict, strict=True)
        if not ok and p.confidence is not None
    ]
    mean = lambda xs: f"{statistics.mean(xs):.2f}" if xs else "-"
    print(f"mean confidence  when right {mean(conf_ok)} | when wrong {mean(conf_bad)}")
    wrong = [i for i, ok in enumerate(strict) if not ok]
    flagged = [
        i
        for i, p in enumerate(first)
        if p.confidence is not None and p.confidence < threshold
    ]
    caught = [i for i in wrong if i in flagged]
    needless = [i for i in flagged if strict[i]]
    print(
        f"handoff at confidence < {threshold}: flags {len(flagged)} | "
        f"catches {pct(len(caught), len(wrong))} of wrong answers | "
        f"needlessly flags {pct(len(needless), sum(strict))} of right ones"
    )

    # stability across repeats
    if len(runs) > 1:
        flips = sum(len({run[i].label for run in runs}) > 1 for i in range(len(rows)))
        print(
            f"label changed between {len(runs)} repeats on {flips}/{len(rows)} questions"
        )

    latencies = [p.latency_s for run in runs for p in run if not p.error]
    cost = sum(p.cost_usd for run in runs for p in run)
    calls = len(rows) * len(runs)
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(f"latency median {statistics.median(latencies):.2f}s | p95 {p95:.2f}s")
    print(f"cost ${cost:.5f} for {calls} calls (${cost / calls:.6f} each)")

    print("\nwrong answers (strict):")
    for i in wrong:
        r, p = rows[i], first[i]
        verdict = "lenient ok" if lenient[i] else "WRONG     "
        conf = f"{p.confidence:.2f}" if p.confidence is not None else " - "
        print(
            f"  {verdict} {r['id']} gold={r['label']:11s} got={p.label!s:11s} conf={conf} | {r['question'][:52]}"
        )

    confusion: dict[str, Counter] = defaultdict(Counter)
    for r, p in zip(rows, first, strict=True):
        confusion[r["label"]][str(p.label)] += 1
    return {
        "name": name,
        "strict": sum(strict),
        "lenient": sum(lenient),
        "n": len(rows),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "predictions": [[asdict(p) for p in run] for run in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--llm-model", default=get_settings().summary_model)
    args = parser.parse_args()

    key = get_settings().openrouter_api_key
    if not key:
        sys.exit("no OPENROUTER_API_KEY in settings")

    rows = load_questions()
    print(
        f"{len(rows)} questions | repeats {args.repeats} | handoff threshold {args.threshold}"
    )
    candidates: list[Classifier] = [
        JevClassifier(key),
        LLMClassifier(key, args.llm_model),
    ]

    results = []
    for classifier in candidates:
        results.append(
            report(
                classifier.name,
                rows,
                run(classifier, rows, args.repeats),
                args.threshold,
            )
        )

    print(f"\n{'=' * 78}\nHEAD TO HEAD (first run)\n{'=' * 78}")
    for res in results:
        print(
            f"{res['name']:32s} strict {pct(res['strict'], res['n']):12s} lenient {pct(res['lenient'], res['n'])}"
        )

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now(tz=UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nsaved {out.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
