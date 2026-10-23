"""Build staged-outcome samples (one per case) for evaluation and RL.

  python dataset/build_samples.py --cases xgb_test_283.csv --split test --out runs/test_samples
  python dataset/build_samples.py --cases train_clean.csv --split train --fold 0 --out runs/dev_fold0

For every case it writes <out>/cases/<case_id>/:
  prompt.md      the staged case text the agent sees (no case id, no outcome)
  metadata.json  outcome label and the true tool inputs, read by the grader after the agent
  service.env    settings for start_services.sh (fold-held-out tools for training cases)
and <out>/samples.jsonl in the Dressage claw format (one line per case).

Splits
  test   cases outside the 1132-case training set: tools use all training cases and the full model.
  train  training cases (RL, or a development fold with --fold): the similar-patient tool hides the
         case's fold and the prediction tool uses the model trained without that fold.
The case CSV is cleaned with the same outcome-blind rules as the tools. Patient data and labels stay
in <out>, which must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "task" / "services"))

from clinical_tools.cleaning import load_training, rows_to_training  # noqa: E402
from clinical_tools.common import FEATURE_NAMES, assign_folds  # noqa: E402

PUBLIC_TRAIN = REPO / "task" / "services" / "data" / "train_age_amh_gn.csv"
TEMPLATE_BASE = "/eds-data/home/kuisu/03.projects/03.agent/Dressage/examples/data/dressage_claw_sample/tasks"
OUTCOME = {1: "妊娠", 0: "未妊娠"}
EMBRYO_COLUMN = "胚胎冷冻数"          # optional column; its definition is unconfirmed and it may leak the outcome

# (label shown to the agent, column, unit) per stage group
STAGE1_FEMALE = [("年龄", "女方年龄", "岁"), ("AMH", "女方AMH", "ng/mL"), ("基础FSH", "女方基础FSH", ""),
                 ("PRL", "女方基础PRL", ""), ("T", "女方基础T", ""), ("P", "女方基础P", ""),
                 ("白细胞", "女方白细胞计数", ""), ("血小板", "女方血小板", ""), ("血细胞压积", "女方血细胞压积", ""),
                 ("PT", "女方凝血常规PT", ""), ("APTT", "女方凝血常规APPT", ""), ("TT", "女方凝血常规TT", ""),
                 ("D二聚体", "女方D二聚体", ""), ("甲功TPO", "女方甲功TPO", ""), ("空腹血糖", "女方空腹血糖", ""),
                 ("尿酸", "女方肾功尿酸", "")]
STAGE1_MALE = [("精液浓度", "男方浓度", ""), ("PR", "男方PR", ""), ("NP", "男方NP", ""),
               ("正常形态率", "男方正常形态率", ""), ("存活率", "男方存活率", ""), ("DFI", "男方DFI", ""),
               ("ALT", "男方ALT", ""), ("AST", "男方AST", ""), ("血肌酐", "男方血肌酐", ""),
               ("白细胞", "男方白细胞", ""), ("血小板", "男方血小板", ""), ("空腹血糖", "男方空腹血糖", "")]
STAGE2 = [("Gn用药", "女方用药Gn总天数", "天"), ("HCG日P", "女方用药HCG日P", ""),
          ("处理后NP", "男方精液处理后NP", ""), ("处理后密度", "男方精液处理后密度", ""),
          ("处理后PR", "男方精液处理后PR", ""), ("处理后IM", "男方精液处理后IM", "")]
assert {c for _, c, _ in STAGE1_FEMALE + STAGE1_MALE + STAGE2} | {"男方精液处理方法"} == set(FEATURE_NAMES)


def _show(value: float | None, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "未提供"
    return f"{value:g}{(' ' + unit) if unit else ''}"


def _line(items, values) -> str:
    return "，".join(f"{label} {_show(values.get(col), unit)}" for label, col, unit in items)


def format_case(values: dict[str, float | None], embryo: str | None = None) -> str:
    """Staged case text for the agent: no case id, no outcome."""
    method = values.get("男方精液处理方法")
    method_text = "未提供" if method is None or math.isnan(method) else f"编码 {method:g}（含义未确认）"
    embryo_text = f"胚胎冷冻数 {embryo}（定义未确认）。" if embryo not in (None, "") else "未提供胚胎信息。"
    return "\n".join([
        "以下是一位患者的研究数据，请按三个阶段分析，最后判断本次新鲜胚胎移植的临床妊娠结局。",
        "",
        "【初诊：年龄与周期前检查】",
        f"女方：{_line(STAGE1_FEMALE, values)}。",
        f"男方：{_line(STAGE1_MALE, values)}。",
        "",
        "【促排卵与取卵】",
        f"{_line(STAGE2, values)}；精液处理方法 {method_text}。",
        "",
        "【胚胎】",
        embryo_text,
    ]) + "\n"


def _number(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def _task_files(task_root: str) -> tuple[list[dict], list[dict]]:
    """Task and grader files per task.json's mapping (git-tracked files only)."""
    task = json.loads((REPO / "task.json").read_text(encoding="utf-8"))
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.split()

    def mapped(source_dir: str, target: str) -> list[dict]:
        prefix = source_dir.rstrip("/") + "/"
        return [{"path": f"{target}/{p[len(prefix):]}", "local_path": f"{task_root}/{p}"}
                for p in sorted(tracked) if p.startswith(prefix) and not p.endswith(".gitkeep")]

    before = [f for e in task["files"]["task"] if e["source"] != "workspace"
              for f in mapped(f"task/{e['source']}", e["target"])]
    after = [f for e in task["grader"]["files"] for f in mapped(e["source"], e["target"])]
    return before, after


def _commands(service_env: dict[str, str]) -> dict:
    env = " ".join(f"{k}={v}" for k, v in service_env.items())
    oc_ws = 'OC_WS=$(ls -d "${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}"/bbs-*/home/.openclaw/workspace 2>/dev/null | head -1)'
    return {
        "before_agent": [
            {"name": "map_workspace", "cmd": f'mkdir -p /workspace; {oc_ws}; if [ -n "$OC_WS" ] && [ -n "$(ls -A /workspace 2>/dev/null)" ]; then cp -r /workspace/* "$OC_WS/"; fi; ls -la /workspace/', "required": True, "timeout": 15.0},
            {"name": "setup_task", "cmd": "bash /tmp/omni_task/scripts/setup.sh", "required": True, "timeout": 120.0},
            {"name": "start_services", "cmd": f"{env} bash /tmp/omni_task/scripts/start_services.sh".strip(), "required": True, "timeout": 90.0},
            {"name": "register_mcp", "cmd": 'OC_HOME=$(ls -d "${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}"/bbs-*/home 2>/dev/null | head -1); HOME="$OC_HOME" openclaw mcp set claw-tools \'{"command":"python3","args":["/tmp/omni_task/runtime/claw_tools_mcp.py"],"env":{"OMNI_TOOL_SPECS_DIR":"/tmp/omni_task/tools","OMNI_TRACE_DIR":"/tmp/omni_task/trace","NO_PROXY":"127.0.0.1,localhost,0.0.0.0,::1","no_proxy":"127.0.0.1,localhost,0.0.0.0,::1"},"cwd":"/tmp/omni_task"}\'', "required": True, "timeout": 120.0},
            {"name": "expose_tool_instructions", "cmd": "OMNI_TOOL_SPECS_DIR=/tmp/omni_task/tools python3 /tmp/omni_task/runtime/claw_tools_mcp.py --append-agents-md", "required": True, "timeout": 15.0},
        ],
        "after_agent": [
            {"name": "patch_llm_judge", "cmd": "python3 /tmp/omni_task/runtime/patch_llm_judge.py /tmp/omni_grader", "required": True, "timeout": 30.0},
            {"name": "generate_trace", "cmd": "python3 /tmp/omni_task/runtime/gen_trace.py", "required": True, "timeout": 15.0},
            {"name": "stop_services", "cmd": "bash /tmp/omni_task/scripts/stop_services.sh", "required": True, "timeout": 30.0},
            {"name": "run_grader", "cmd": f'{oc_ws}; [ -n "$OC_WS" ] && cp -r "$OC_WS/"* /workspace/ 2>/dev/null; bash /tmp/omni_grader/run.sh', "required": True, "timeout": 120.0},
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True, help="case CSV (id, 35 indicators, label)")
    parser.add_argument("--split", choices=("test", "train"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), help="train split only: keep cases of this fold")
    parser.add_argument("--limit", type=int, help="keep the first N cases (after the fold filter)")
    parser.add_argument("--out", type=Path, required=True, help="new output directory (holds labels; never commit)")
    parser.add_argument("--task-root", default=None,
                        help="where the dataset machine keeps this repository (local_path of task files)")
    parser.add_argument("--cases-root", default=None,
                        help="where the dataset machine keeps <out>/cases (local_path of metadata.json)")
    args = parser.parse_args(argv)

    import csv
    with open(args.cases, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cases = rows_to_training(rows)                   # outcome-blind cleaning, as in the tools
    embryo = {r["id"]: r.get(EMBRYO_COLUMN, "") for r in rows}

    train = load_training(PUBLIC_TRAIN)
    folds = dict(zip(train.case_ids, assign_folds(train.case_ids, train.labels.tolist(), 5)))
    if args.split == "train":
        unknown = [c for c in cases.case_ids if c not in folds]
        if unknown:
            raise SystemExit(f"{len(unknown)} cases are not training cases (e.g. {unknown[:3]}); use --split test")
    else:
        overlap = [c for c in cases.case_ids if c in folds]
        if overlap:
            raise SystemExit(f"{len(overlap)} cases are training cases (e.g. {overlap[:3]}); use --split train")
        if args.fold is not None:
            raise SystemExit("--fold applies to the train split only")

    selected = [i for i, c in enumerate(cases.case_ids) if args.fold is None or folds.get(c) == args.fold]
    # Hash order, not file order: in this dataset the id order alone separates the outcomes.
    selected.sort(key=lambda i: hashlib.sha256(f"sample:{cases.case_ids[i]}".encode()).hexdigest())
    if args.limit:
        selected = selected[:args.limit]
    args.out.mkdir(parents=True, exist_ok=False)

    task_id = "ivf-clinical-tools_" + hashlib.sha256(
        json.loads((REPO / "task.json").read_text(encoding="utf-8"))["id"].encode()).hexdigest()[:12]
    task_root = args.task_root or f"{TEMPLATE_BASE}/{task_id}"
    cases_root = args.cases_root or str((args.out / "cases").resolve())
    before, after = _task_files(task_root)
    column = {name: i for i, name in enumerate(FEATURE_NAMES)}

    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as jsonl:
        for i in selected:
            case_id = cases.case_ids[i]
            values = {name: float(cases.matrix[i, column[name]]) for name in FEATURE_NAMES}
            prompt = format_case(values, embryo.get(case_id))
            service_env: dict[str, str] = {}
            fold = folds.get(case_id)
            if args.split == "train":
                service_env = {"IVF_EXCLUDE_CASE": case_id, "IVF_MODEL_NAME": f"lr_35_indicators_fold{fold}.json"}
            metadata = {
                "case_id": case_id, "split": args.split, "fold": fold,
                "label": OUTCOME[int(cases.labels[i])],
                "inputs": {"age": _number(values["女方年龄"]), "amh": _number(values["女方AMH"]),
                           "gn_days": _number(values["女方用药Gn总天数"])},
            }
            case_dir = args.out / "cases" / case_id
            case_dir.mkdir(parents=True)
            (case_dir / "prompt.md").write_text(prompt, encoding="utf-8")
            (case_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
            (case_dir / "service.env").write_text("".join(f"{k}={v}\n" for k, v in service_env.items()),
                                                  encoding="utf-8")
            sample = {
                "prompt": prompt, "label": "", "reward_fn": "claw_grader",
                "metadata": {
                    "instance_id": f"claw-ivf-staged-{args.split}-{case_id}", "task_id": task_id,
                    "blackbox_type": "openclaw", "sandbox_image": "e2b-dressage-claw-blackbox",
                    "sandbox_timeout_sec": 3600, "system_prompt_file": "/tmp/omni_task/runtime/system_prompt_prefix.txt",
                    "before_agent_files": before,
                    "after_agent_files": after + [{"path": "/tmp/omni_grader/metadata.json",
                                                   "local_path": f"{cases_root}/{case_id}/metadata.json"}],
                    "blackbox_execute_cmds": _commands(service_env),
                },
            }
            jsonl.write(json.dumps(sample, ensure_ascii=False) + "\n")
    labels = [int(cases.labels[i]) for i in selected]
    print(json.dumps({"out": str(args.out), "split": args.split, "fold": args.fold, "cases": len(selected),
                      "pregnant": sum(labels), "task_id": task_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
