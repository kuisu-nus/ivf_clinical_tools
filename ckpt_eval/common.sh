# Shared settings for the ckpt_eval scripts (sourced, not run). Override any of these in the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$REPO/ckpt_eval"
PYTHON="${PYTHON:-python3}"
DRESSAGE_ROOT="${DRESSAGE_ROOT:-$HOME/Dressage}"            # Dressage checkout (slime submodule inside)
MEGATRON_ROOT="${MEGATRON_ROOT:-$HOME/Megatron-LM}"         # Megatron-LM matching slime (see rl/README.md)
OC_SOURCE_HOME="${OC_SOURCE_HOME:-$HOME}"                   # whose ~/.openclaw is copied / deployed to
OC_EVAL_ROOT="${OC_EVAL_ROOT:-$OC_SOURCE_HOME/oc_eval}"     # one isolated OpenClaw home per model
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-$OC_SOURCE_HOME/sglang_logs}"
OC_PROVIDER="${OC_PROVIDER:-ivf}"                           # provider id the local models are registered under

die() { echo "error: $*" >&2; exit 1; }

# GET http://127.0.0.1:<port>/v1/models without any proxy; prints the body (empty when nothing answers).
models_at() { curl -s --noproxy '*' --max-time 3 "http://127.0.0.1:$1/v1/models" 2>/dev/null || true; }
