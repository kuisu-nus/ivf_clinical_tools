"""Shared V4 grader runner.

Task authors should not edit or copy this logic. A task directory only holds
``grader.py`` (the scoring logic) plus a thin ``run.sh`` that sets PYTHONPATH and
invokes this shared runner; there is no per-task ``run_grader.py``. All context
construction, optional LLM judge dispatch, validation, aggregation, and output
formatting live here so every task scores through one code path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omni_grader_helpers import aggregate_score_items, validate_score_items


def _load_grader_module(grader_dir: Path):
    grader_path = grader_dir / "grader.py"
    spec = importlib.util.spec_from_file_location("omni_task_grader", grader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load grader: {grader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path_from_env(name: str, default: Path | str) -> Path:
    return Path(os.environ.get(name) or default)


def _default_workspace(argv: list[str] | None = None) -> Path:
    if os.environ.get("OMNI_WORKSPACE_PATH"):
        return Path(os.environ["OMNI_WORKSPACE_PATH"])
    args = sys.argv if argv is None else argv
    if len(args) > 1:
        return Path(args[1])
    if Path("/workspace").exists():
        return Path("/workspace")
    return Path("/root")


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_ctx(
    *,
    grader_dir: Path,
    argv: list[str] | None = None,
) -> SimpleNamespace:
    grader_mount = _path_from_env("OMNI_GRADER_MOUNT_PATH", grader_dir)
    metadata_path = _path_from_env("OMNI_GRADER_METADATA_PATH", grader_mount / "metadata.json")
    return SimpleNamespace(
        task_id=os.environ.get("OMNI_TASK_ID", ""),
        workspace_path=_default_workspace(argv),
        grader_mount_path=grader_mount,
        tool_trajectory_path=_path_from_env("OMNI_TOOL_TRAJECTORY_PATH", grader_mount / "tool_trajectory.json"),
        llm_messages_path=_path_from_env("OMNI_LLM_MESSAGES_PATH", grader_mount / "llm_messages.json"),
        audit_data_path=_path_from_env("OMNI_AUDIT_DATA_PATH", grader_mount / "audit_data.json"),
        grader_assets_path=_path_from_env("OMNI_GRADER_ASSETS_PATH", grader_mount / "assets"),
        metadata=_load_metadata(metadata_path),
    )


def maybe_judge() -> Any:
    """Return the injected judge client, if this runtime provides one.

    If no runtime judge is configured, shared ``judge_rubric`` emits skipped LLM
    items so aggregation reports ``partial_evaluation`` instead of a false
    complete score. Real judge construction belongs to the runtime, not task
    graders.
    """
    flag = str(os.environ.get("OMNI_JUDGE_ENABLE", "")).strip().lower()
    claw_flag = str(os.environ.get("OMNI_CLAW_EVAL_ENABLE_JUDGE", "")).strip().lower()
    if flag not in {"1", "true", "yes", "on"} and claw_flag not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        from omni_judge_runtime import adapt_for_claw_eval, build_judge_from_env
    except ImportError:
        return None
    return adapt_for_claw_eval(build_judge_from_env())


def legacy_overall_score(aggregation: dict[str, Any]) -> float:
    """Return the numeric score consumed by legacy reward paths.

    ``available_score`` is normalized over only the evaluated score weight, so
    a hybrid task with deterministic half fully passing and LLM half skipped has
    ``available_score == 1.0``. Legacy reward consumers need a conservative
    task-level number instead: skipped weight contributes zero.
    """
    available_score = aggregation.get("available_score")
    if not isinstance(available_score, (int, float)) or isinstance(available_score, bool):
        return 0.0
    scored_weight = aggregation.get("scored_weight")
    total_weight = aggregation.get("total_declared_weight")
    if (
        isinstance(scored_weight, (int, float))
        and not isinstance(scored_weight, bool)
        and isinstance(total_weight, (int, float))
        and not isinstance(total_weight, bool)
        and total_weight > 0
    ):
        return float(available_score) * float(scored_weight) / float(total_weight)
    return float(available_score)


def run_grader(
    *,
    grader_dir: Path,
    judge: Any = None,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    module = _load_grader_module(grader_dir)
    ctx = build_ctx(grader_dir=grader_dir, argv=argv)

    grade = getattr(module, "grade", None)
    if grade is None:
        raise RuntimeError("grader.py does not define grade(ctx)")

    items = list(grade(ctx))
    validate_score_items(items, function_name="grade")

    llm_judge = getattr(module, "llm_judge", None)
    if callable(llm_judge):
        llm_items = list(llm_judge(ctx, maybe_judge() if judge is None else judge))
        validate_score_items(llm_items, function_name="llm_judge")
        items += llm_items

    aggregation = aggregate_score_items(items)
    available_score = aggregation.get("available_score")
    overall_score = legacy_overall_score(aggregation)
    evaluation_status = str(aggregation.get("evaluation_status") or aggregation.get("status") or "error")
    outcome_status = str(aggregation.get("outcome_status") or "unknown")
    official_score = None if evaluation_status in {"partial_evaluation", "skipped", "error"} else available_score

    return {
        "overall_score": overall_score,
        "legacy_reward_score": overall_score,
        "available_score": available_score,
        "official_score": official_score,
        "evaluation_status": evaluation_status,
        "outcome_status": outcome_status,
        "scores": [
            {
                "name": "automated_checks",
                "type": "command",
                "score": overall_score,
                "weight": 1.0,
            }
        ],
        "raw_scores": {
            "items": items,
            "aggregation": aggregation,
        },
    }


def main(
    *,
    grader_dir: str | Path | None = None,
    judge: Any = None,
    argv: list[str] | None = None,
) -> None:
    resolved_grader_dir = (
        Path(grader_dir)
        if grader_dir is not None
        else Path(os.environ.get("OMNI_GRADER_DIR") or Path(__file__).resolve().parent)
    )
    print(
        json.dumps(
            run_grader(grader_dir=resolved_grader_dir, judge=judge, argv=argv),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
