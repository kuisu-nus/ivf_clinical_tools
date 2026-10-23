"""Summarise a local evaluation: outcome metrics, reward components and answer length.

  python dataset/summarize_eval.py <eval_dir>      (writes <eval_dir>/summary.json and prints a table)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean


def main(argv: list[str] | None = None) -> None:
    eval_dir = Path((argv or sys.argv[1:])[0])
    rows = []
    for path in sorted(eval_dir.glob("cases/*/score.json")):
        score = json.loads(path.read_text(encoding="utf-8"))
        items = {i["name"]: i for i in score["grader"]["raw_scores"]["items"]}
        outcome = items.get("outcome", {}).get("metadata", {})
        rows.append({"reward": score["grader"]["overall_score"], "predicted": outcome.get("predicted"),
                     "truth": outcome.get("truth"), "gate_failed": items["no_treatment_advice"]["status"] == "failed",
                     **{k: items[k]["score"] for k in ("outcome", "tool_use", "faithfulness", "format", "concise")
                        if k in items},
                     "characters": items.get("concise", {}).get("metadata", {}).get("characters"),
                     "output_tokens": score["run"].get("output_tokens"), "duration_ms": score["run"].get("duration_ms")})
    if not rows:
        raise SystemExit(f"no scored cases in {eval_dir}")

    def share(cond) -> float:
        return round(sum(1 for r in rows if cond(r)) / len(rows), 3)

    pos = [r for r in rows if r["truth"] == "妊娠"]
    neg = [r for r in rows if r["truth"] == "未妊娠"]
    decided = [r for r in rows if r["predicted"] in ("妊娠", "未妊娠")]
    sens = sum(r["predicted"] == "妊娠" for r in pos) / len(pos) if pos else None
    spec = sum(r["predicted"] == "未妊娠" for r in neg) / len(neg) if neg else None
    summary = {
        "cases": len(rows), "pregnant": len(pos), "not_pregnant": len(neg),
        "answers": {a: sum(r["predicted"] == a for r in rows) for a in ("妊娠", "未妊娠", "不确定", None)},
        "accuracy_all": share(lambda r: r["predicted"] == r["truth"]),
        "accuracy_decided": round(sum(r["predicted"] == r["truth"] for r in decided) / len(decided), 3) if decided else None,
        "sensitivity": None if sens is None else round(sens, 3),
        "specificity": None if spec is None else round(spec, 3),
        "balanced_accuracy": None if sens is None or spec is None else round((sens + spec) / 2, 3),
        "uncertain_rate": share(lambda r: r["predicted"] == "不确定"),
        "no_conclusion_rate": share(lambda r: r["predicted"] is None),
        "gate_failed_rate": share(lambda r: r["gate_failed"]),
        "mean_reward": round(mean(r["reward"] for r in rows), 4),
        "mean_component": {k: round(mean(r[k] for r in rows if r.get(k) is not None), 4)
                           for k in ("outcome", "tool_use", "faithfulness", "format", "concise")},
        "mean_answer_characters": round(mean(r["characters"] for r in rows if r["characters"] is not None), 1),
        "mean_output_tokens": round(mean(r["output_tokens"] for r in rows if r["output_tokens"]), 1)
        if any(r["output_tokens"] for r in rows) else None,
        "note": "No-skill policies (always one answer) score an expected outcome 0.5; balanced accuracy 0.5 is chance.",
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
