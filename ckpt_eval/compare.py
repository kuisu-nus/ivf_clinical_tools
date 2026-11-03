"""Compare models evaluated on the same sample set (ckpt_eval/run_eval.sh or openclaw_eval/run_all.sh).

  python ckpt_eval/compare.py <eval_dir> [<eval_dir> ...] [--names step0 step100 ...] [--export rows.jsonl]

Prints a Markdown table per model (answers, balanced accuracy, how often each answer is right, mean
reward and its parts, tool calls) and how the models' conclusions combine case by case. --export writes
one JSON line per model and case (answer text, scores and the grader's reasons) for reading the answers;
it holds labels, so keep it out of the repository. Nothing printed names a case.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

PARTS = ("outcome", "tool_use", "faithfulness", "format", "concise")
ANSWERS = ("妊娠", "未妊娠", "不确定")


def load(eval_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for score_file in sorted(eval_dir.glob("cases/*/score.json")):
        case = score_file.parent
        grader = json.loads(score_file.read_text(encoding="utf-8"))["grader"]
        items = {i["name"]: i for i in grader["raw_scores"]["items"]}
        outcome = items.get("outcome", {}).get("metadata", {})
        dispatches = case / "dispatches.jsonl"
        answer = case / "answer.txt"
        rows[case.name] = {
            "predicted": outcome.get("predicted"), "truth": outcome.get("truth"),
            "reward": grader["overall_score"],
            "scores": {k: v["score"] for k, v in items.items()},
            "reasons": {k: v.get("reason") for k, v in items.items()},
            "tool_calls": sum(1 for line in dispatches.read_text(encoding="utf-8").splitlines() if line.strip())
            if dispatches.exists() else None,
            "answer": answer.read_text(encoding="utf-8").strip() if answer.exists() else None,
        }
    return rows


def metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r = list(rows.values())
    pos = [x for x in r if x["truth"] == "妊娠"]
    neg = [x for x in r if x["truth"] == "未妊娠"]
    sens = sum(x["predicted"] == "妊娠" for x in pos) / len(pos) if pos else float("nan")
    spec = sum(x["predicted"] == "未妊娠" for x in neg) / len(neg) if neg else float("nan")

    def right_when(answer: str) -> str:
        said = [x for x in r if x["predicted"] == answer]
        return f"{sum(x['truth'] == answer for x in said)}/{len(said)}" if said else "-"

    calls = [x["tool_calls"] for x in r if x["tool_calls"] is not None]
    return {"cases": len(r), "answers": " / ".join(str(sum(x["predicted"] == a for x in r)) for a in ANSWERS),
            "no_conclusion": sum(x["predicted"] not in ANSWERS for x in r),
            "balanced_accuracy": round((sens + spec) / 2, 3), "sensitivity": round(sens, 3),
            "specificity": round(spec, 3), "right_when_妊娠": right_when("妊娠"),
            "right_when_未妊娠": right_when("未妊娠"), "mean_reward": round(mean(x["reward"] for x in r), 3),
            **{p: round(mean(x["scores"][p] for x in r if x["scores"].get(p) is not None), 3)
               for p in PARTS if any(x["scores"].get(p) is not None for x in r)},
            "tool_calls": round(mean(calls), 2) if calls else None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_dirs", nargs="+", type=Path)
    parser.add_argument("--names", nargs="+", help="column names (default: each eval dir's parent name)")
    parser.add_argument("--export", type=Path, help="write every answer with its scores to this JSONL file")
    args = parser.parse_args(argv)
    names = args.names or [d.resolve().parent.name for d in args.eval_dirs]
    if len(names) != len(args.eval_dirs):
        parser.error("--names needs one name per eval dir")
    data = {n: load(d) for n, d in zip(names, args.eval_dirs)}
    for n, rows in data.items():
        if not rows:
            raise SystemExit(f"no scored cases for {n}")

    table = {n: metrics(rows) for n, rows in data.items()}
    keys = list(next(iter(table.values())))
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    labels = {"answers": "answers 妊娠 / 未妊娠 / 不确定"}
    for k in keys:
        print(f"| {labels.get(k, k)} | " + " | ".join(str(table[n].get(k)) for n in names) + " |")

    common = set.intersection(*(set(rows) for rows in data.values()))
    if len(names) > 1 and common:
        combos = Counter(tuple(data[n][c]["predicted"] for n in names) for c in common)
        print(f"\nconclusions per case ({' / '.join(names)}), {len(common)} cases in all:")
        for combo, count in combos.most_common(8):
            print(f"  {count:4d}  {' / '.join(str(a) for a in combo)}")

    if args.export:
        with args.export.open("w", encoding="utf-8") as out:
            for n, rows in data.items():
                for case, row in rows.items():
                    out.write(json.dumps({"model": n, "case": case, **row}, ensure_ascii=False) + "\n")
        print(f"\nexported {sum(len(r) for r in data.values())} rows to {args.export} (holds labels: do not commit)")


if __name__ == "__main__":
    main()
