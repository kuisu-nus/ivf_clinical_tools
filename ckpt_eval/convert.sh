#!/usr/bin/env bash
# 1. Model conversion: a slime/Megatron torch_dist checkpoint (iter_XXXXXXX) -> Hugging Face weights,
#    then check that the converted weight names match the original model exactly.
#
#   bash ckpt_eval/convert.sh <iter_dir> <out_dir> <origin_hf_dir>
#   e.g. bash ckpt_eval/convert.sh ~/Dressage/models/Qwen3-4B-Instruct-2507_slime/iter_0000099 \
#          ~/Dressage/models/Qwen3-4B-ivf-step100-hf ~/Dressage/models/Qwen3-4B-Instruct-2507
#
# Runs on CPU (a 4B model takes a few minutes). Needs DRESSAGE_ROOT/slime and MEGATRON_ROOT.
set -euo pipefail
source "$(dirname "$0")/common.sh"
ITER="${1:?usage: convert.sh <iter_dir> <out_dir> <origin_hf_dir>}"
OUT="${2:?usage: convert.sh <iter_dir> <out_dir> <origin_hf_dir>}"
ORIGIN="${3:?usage: convert.sh <iter_dir> <out_dir> <origin_hf_dir>}"
[ -f "$ITER/common.pt" ] || die "$ITER is not a torch_dist checkpoint (no common.pt)"
[ -f "$ORIGIN/config.json" ] || die "$ORIGIN is not a Hugging Face model (no config.json)"
[ -e "$OUT" ] && die "$OUT already exists"
[ -f "$DRESSAGE_ROOT/slime/tools/convert_torch_dist_to_hf.py" ] || die "slime not found under $DRESSAGE_ROOT/slime"
[ -d "$MEGATRON_ROOT/megatron" ] || die "Megatron-LM not found at $MEGATRON_ROOT (set MEGATRON_ROOT)"

(cd "$DRESSAGE_ROOT/slime" && CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH="$MEGATRON_ROOT:$DRESSAGE_ROOT/slime${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" tools/convert_torch_dist_to_hf.py --input-dir "$ITER" --output-dir "$OUT" --origin-hf-dir "$ORIGIN" -a) \
  2>&1 | tail -n 3

"$PYTHON" "$HERE/check_weights.py" "$ORIGIN" "$OUT"
