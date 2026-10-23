#!/usr/bin/env bash
# Stage 3 (after_agent): score the answer with the task grader.
#
#   bash openclaw_eval/after_agent.sh <case_dir> <result_dir>
#
# As in the Dressage sandbox, the grader and metadata.json (the outcome label) are put next to
# the answer only now, after the agent has finished. Steps:
#   grader files + metadata.json      -> <result_dir>/grader_mount/
#   prompt + answer.txt               -> grader_mount/agent_messages.json
#   gen_trace.py (dispatches.jsonl)   -> grader_mount/tool_trajectory.json, llm_messages.json
#   grader run.sh                     -> <result_dir>/grader_output.json (overall_score and items)
# Also writes <result_dir>/score.json (grader output + run statistics, read by summarize_eval.py).
set -euo pipefail
source "$(dirname "$0")/common.sh"
CASE="$(cd "${1:?usage: after_agent.sh <case_dir> <result_dir>}" && pwd)"
RES="$(cd "${2:?usage: after_agent.sh <case_dir> <result_dir>}" && pwd)"
[ -f "$RES/answer.txt" ] || die "no answer.txt in $RES (run agent.sh first)"
G="$RES/grader_mount"
rm -rf "$G"; mkdir -p "$G"

cp "$REPO"/grader/*.py "$REPO/grader/run.sh" "$G/"
cp "$CASE/metadata.json" "$G/metadata.json"
"$PYTHON" - "$CASE/prompt.md" "$RES/answer.txt" "$G/agent_messages.json" <<'PY'
import json, sys
prompt_file, answer_file, out = sys.argv[1:]
prompt, answer = (open(p, encoding="utf-8").read() for p in (prompt_file, answer_file))
json.dump({"prompt": prompt, "response": answer.strip(), "messages": []}, open(out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
PY
OMNI_GRADER_DIR="$G" OMNI_TRACE_DIR="$RES" "$PYTHON" "$REPO/task/runtime/gen_trace.py" > "$RES/after_agent.log"
OMNI_GRADER_MOUNT_PATH="$G" bash "$G/run.sh" > "$RES/grader_output.json" 2>> "$RES/after_agent.log" \
  || die "grader failed; see $RES/after_agent.log"

"$PYTHON" - "$RES" "$(basename "$CASE")" <<'PY'
import json, sys
from pathlib import Path
res, case_id = Path(sys.argv[1]), sys.argv[2]
graded = json.loads((res / "grader_output.json").read_text(encoding="utf-8"))
run = json.loads((res / "agent_run.json").read_text()) if (res / "agent_run.json").exists() else {}
(res / "score.json").write_text(json.dumps({"case_id": case_id, "run": run, "grader": graded},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
items = {i["name"]: i for i in graded["raw_scores"]["items"]}
outcome = items.get("outcome", {}).get("metadata", {})
parts = "  ".join(f"{k} {items[k]['score']}" for k in ("outcome", "tool_use", "faithfulness", "format", "concise")
                  if k in items and items[k]["score"] is not None)
print(f"{case_id}: score {graded['overall_score']:.3f}  answer {outcome.get('predicted')}  "
      f"truth {outcome.get('truth')}  | {parts}")
PY
