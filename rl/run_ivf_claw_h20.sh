#!/bin/bash
# GRPO training of the IVF outcome task with OpenClaw on one 8-GPU node (H20), Dressage blackbox +
# local bubblewrap sandboxes. Follows Dressage's docs/blackbox-dressage-claw-experiment-zh.md, with
# the single-node layout of examples/scripts/run_blackbox_qwen3.5_35b_a3b_sync_local.sh:
#   - rollout: dressage.recipes.dressage_claw.dispatch.generate (staged files, before/after commands)
#   - reward:  dressage.recipes.dressage_claw.reward (overall_score printed by run_grader)
#   - sandbox: local_bwrap slots on this machine instead of E2B (no E2B key, H20 is not reachable
#              from the internet); the task's start_services.sh / register_mcp.sh give each slot
#              its own tool-service ports (task/scripts/ports.sh)
#
# Required:
#   PROMPT_DATA   samples.jsonl from dataset/build_samples.py --split train (see rl/README.md)
#   MODEL_ROOT    holds $MODEL_NAME (HF) and ${MODEL_NAME}_torch_dist (converted); checkpoints go to
#                 ${MODEL_NAME}_ivf_<RUN_NAME>/
# Optional (defaults in brackets):
#   DRESSAGE_ROOT [~/Dressage]  MEGATRON_ROOT [/root/Megatron-LM if readable, else ~/Megatron-LM]
#   MODEL_NAME [Qwen3.6-35B-A3B]  MODEL_CONFIG [qwen3.6-35B-A3B.sh, else qwen3.5-35B-A3B.sh]  TOKEN_BUILD_MODEL [qwen3_6]
#   RUN_NAME [ivf-claw-<date>]  NUM_ROLLOUT [100]  ROLLOUT_BATCH_SIZE [8]  N_SAMPLES_PER_PROMPT [8]
#   ROLLOUT_MAX_RESPONSE_LEN [4096]  CP_SIZE [4 -> 32k context]  SAVE_INTERVAL [10]  LR [1e-6]
#   DRESSAGE_BLACKBOX_SLOTS_PER_NODE [16]  DRESSAGE_BLACKBOX_MAX_STEPS [20]  ROLLOUT_TEMPERATURE [1.0]
#
# Smoke test first (2 steps, 4 cases x 4 samples):
#   NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 PROMPT_DATA=... MODEL_ROOT=... bash rl/run_ivf_claw_h20.sh
set -ex

DRESSAGE_ROOT="${DRESSAGE_ROOT:-$HOME/Dressage}"
REPO_ROOT="${DRESSAGE_ROOT}"                       # Dressage helpers expect REPO_ROOT = Dressage checkout
SLIME_ROOT="${SLIME_ROOT:-${DRESSAGE_ROOT}/slime}"
# The image installs Megatron-LM under /root, which a non-root account cannot read: then use a
# checkout in $HOME (rl/README.md, step 1).
if [[ -z "${MEGATRON_ROOT:-}" ]]; then
  if [[ -r /root/Megatron-LM/megatron/core/__init__.py ]]; then MEGATRON_ROOT=/root/Megatron-LM; else MEGATRON_ROOT="$HOME/Megatron-LM"; fi
fi
[[ -f "${MEGATRON_ROOT}/megatron/core/__init__.py" ]] || { echo "Megatron-LM not found at ${MEGATRON_ROOT} (rl/README.md, step 1)" >&2; exit 1; }
MODEL_NAME="${MODEL_NAME:-Qwen3.6-35B-A3B}"
# slime (as pinned by Dressage 3e3142f) has no qwen3.6 model file; Qwen3.6-35B-A3B has the same
# architecture as Qwen3.5-35B-A3B (Qwen3_5MoeForConditionalGeneration, same sizes), so that file is used.
MODEL_CONFIG="${MODEL_CONFIG:-qwen3.6-35B-A3B.sh}"
[[ -f "${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}" ]] || MODEL_CONFIG="qwen3.5-35B-A3B.sh"
TOKEN_BUILD_MODEL="${TOKEN_BUILD_MODEL:-qwen3_6}"   # set before the defaults file (it would pick qwen3_5)
RUN_NAME="${RUN_NAME:-ivf-claw-$(date +%Y%m%d-%H%M)}"
: "${PROMPT_DATA:?set PROMPT_DATA to the training samples.jsonl}"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory holding the model checkpoints}"
[[ -f "${PROMPT_DATA}" ]] || { echo "PROMPT_DATA not found: ${PROMPT_DATA}" >&2; exit 1; }
[[ -f "${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}" ]] || {
  echo "missing ${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}; run: cd ${DRESSAGE_ROOT} && git submodule update --init --recursive" >&2; exit 1; }
[[ -d "${MODEL_ROOT}/${MODEL_NAME}" && -d "${MODEL_ROOT}/${MODEL_NAME}_torch_dist" ]] || {
  echo "need ${MODEL_ROOT}/${MODEL_NAME} and ${MODEL_ROOT}/${MODEL_NAME}_torch_dist (rl/README.md, step 2)" >&2; exit 1; }
SAVE_DIR="${SAVE_DIR:-${MODEL_ROOT}/${MODEL_NAME}_ivf_${RUN_NAME}}"

# Stop what an earlier run of this script left behind (this user's processes only; the upstream
# script's `pkill -9 python` would also kill unrelated work on a shared machine).
ray stop --force 2>/dev/null || true
pkill -u "$(id -u)" -f "sglang" 2>/dev/null || true
pkill -u "$(id -u)" -f "dressage.proxy.server" 2>/dev/null || true
sleep 3

export PYTHONBUFFERED=16
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
RAY_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE}"
CP_SIZE="${CP_SIZE:-4}"
SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)

# Before the Dressage defaults, which would otherwise set 100 steps per session.
export DRESSAGE_BLACKBOX_MAX_STEPS="${DRESSAGE_BLACKBOX_MAX_STEPS:-20}"
source "${SLIME_ROOT}/scripts/models/${MODEL_CONFIG}"
source "${DRESSAGE_ROOT}/examples/scripts/default/dressage_env_defaults.sh"
dressage_apply_common_defaults "${RUN_NAME}" "blackbox" "local_bwrap"
dressage_apply_local_bwrap_defaults "${DRESSAGE_BLACKBOX_SLOTS_PER_NODE:-16}"
command -v "${DRESSAGE_BLACKBOX_BWRAP_BIN}" >/dev/null || { echo "missing bubblewrap: ${DRESSAGE_BLACKBOX_BWRAP_BIN}" >&2; exit 1; }
dressage_compute_local_bwrap_resources 0
dressage_validate_proxy_defaults
dressage_clear_trajectory_logs
export PYTHONPATH="${MEGATRON_ROOT}:${REPO_ROOT}:${SLIME_ROOT}:${PYTHONPATH:-}"
dressage_export_common_env
dressage_export_local_bwrap_env
dressage_compute_context_window 8192 "${CP_SIZE}"

# Claw recipe settings (docs/blackbox-dressage-claw-experiment-zh.md), sized for this task:
# about 4 model calls per case (3 tool calls + the answer), answers of at most ~200 characters.
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
COMPACT_RESERVE_TOKENS="${COMPACT_RESERVE_TOKENS:-8192}"
export DRESSAGE_BLACKBOX_COMPACT_THRESHOLD="${DRESSAGE_BLACKBOX_COMPACT_THRESHOLD:-$((CONTEXT_WINDOW - COMPACT_RESERVE_TOKENS))}"
export DRESSAGE_REWARD_MODULES="dressage.recipes.dressage_claw.reward"
export DRESSAGE_PROXY_MAX_STEPS_PER_SESSION="${DRESSAGE_PROXY_MAX_STEPS_PER_SESSION:-20}"
export DRESSAGE_BLACKBOX_BACKEND_TIMEOUT="${DRESSAGE_BLACKBOX_BACKEND_TIMEOUT:-900}"
export DRESSAGE_ROLLOUT_MAX_RETRIES="${DRESSAGE_ROLLOUT_MAX_RETRIES:-0}"
export DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH="${DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH:-0}"
export DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER="${DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER:-2}"

COMM_ARGS=(--rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}")
TRAIN_ENTRY=(python3 train.py)

PROXY_ARGS=(
   --tokenizer-path "${MODEL_ROOT}/${MODEL_NAME}"
   --host "${PROXY_HOST}"
   --port "${PROXY_PORT}"
   --token-build-mode "${TOKEN_BUILD_MODE}"
   --token-build-model "${TOKEN_BUILD_MODEL}"
   "${COMM_ARGS[@]}"
   --context-window "${CONTEXT_WINDOW}"
   --default-max-tokens "${ROLLOUT_MAX_RESPONSE_LEN}"
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/${MODEL_NAME}"
   --ref-load "${MODEL_ROOT}/${MODEL_NAME}_torch_dist"
   --load "${SAVE_DIR}/"
   --save "${SAVE_DIR}/"
   --save-interval "${SAVE_INTERVAL:-10}"
   --no-save-optim
   --no-load-optim
)

ROLLOUT_ARGS=(
   --rollout-function-path dressage.rollout.sync_rollout.generate_rollout_sync
   --custom-generate-function-path dressage.recipes.dressage_claw.dispatch.generate
   --custom-rm-path dressage.reward.custom_rm.custom_rm
   --data-source-path dressage.rollout.data_source.DressageDataSource
   --custom-reward-post-process-path dressage.training.reward_post_process.reward_post_process
   --custom-convert-samples-to-train-data-path dressage.rollout.convert_samples.convert_samples_to_train_data
   --custom-rollout-log-function-path dressage.rollout.log_rollout.log_rollout_data

   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-100}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-$(( ${ROLLOUT_BATCH_SIZE:-8} * ${N_SAMPLES_PER_PROMPT:-8} ))}"
   --balance-data
   --rollout-global-dataset
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size 512
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef "${ENTROPY_COEF:-0.00}"
   --eps-clip "${EPS_CLIP:-0.2}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.5}"
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
   --sglang-log-level warning
   --sglang-chunked-prefill-size 4096
   --sglang-max-prefill-tokens 8192
   --sglang-max-running-requests 64
   --sglang-router-port "${SGLANG_ROUTER_PORT}"
   --router-policy consistent_hashing
)

MISC_ARGS=(
   --custom-config-path "${DRESSAGE_ROOT}/examples/scripts/default/dressage_staleness.yaml"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

cleanup() {
  status=$?
  set +e
  [[ "${DRESSAGE_LOCAL_BWRAP_CLEANUP_ON_EXIT:-1}" == "1" ]] && python -m dressage.sandbox.scripts.stop_local_bwrap
  ray stop --force
  if [[ -f "${PROXY_PID_FILE}" ]]; then kill "$(cat "${PROXY_PID_FILE}")" 2>/dev/null; rm -f "${PROXY_PID_FILE}"; fi
  exit "${status}"
}
trap cleanup EXIT

cd "${REPO_ROOT}"
python3 -m dressage.proxy.server "${PROXY_ARGS[@]}" >"${PROXY_LOG_FILE}" 2>&1 &
echo $! > "${PROXY_PID_FILE}"
for i in $(seq 1 60); do
  curl -sf "${DRESSAGE_PROXY_URL}/health" >/dev/null 2>&1 && break
  [[ "${i}" -eq 60 ]] && { echo "Dressage proxy failed health check; see ${PROXY_LOG_FILE}" >&2; exit 1; }
  sleep 1
done

export no_proxy="127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR},${PROXY_PUBLIC_HOST},${SGLANG_ROUTER_HOST}"
export NO_PROXY="${no_proxy}"
cd "${SLIME_ROOT}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 "${DRESSAGE_BLACKBOX_HEAD_RESOURCE_ARGS[@]}"
python -m dressage.sandbox.scripts.start_local_bwrap

export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}" TP_SOCKET_IFNAME="${SOCKET_IFNAME}" MASTER_ADDR
export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE="${HAS_NVLINK}"
RUNTIME_ENV_JSON="$(python3 -c 'import json, os
keys = ["no_proxy", "NO_PROXY", "MASTER_ADDR", "PYTHONPATH", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
        "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "WANDB_API_KEY", "WANDB_MODE",
        "DRESSAGE_PROXY_URL", "DRESSAGE_PADDOCK_MODE", "DRESSAGE_SANDBOX_PROVIDER",
        "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", "DRESSAGE_REWARD_MODULES",
        "DRESSAGE_BLACKBOX_MAX_STEPS", "DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "DRESSAGE_PROXY_MAX_STEPS_PER_SESSION",
        "DRESSAGE_BLACKBOX_BACKEND_TIMEOUT", "DRESSAGE_ROLLOUT_MAX_RETRIES", "DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH",
        "DRESSAGE_SYNC_FAILED_GROUP_REPLACEMENT_MULTIPLIER",
        "DRESSAGE_LOCAL_BWRAP_POOL_MODE", "DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE", "DRESSAGE_LOCAL_BWRAP_MANAGER_NAME",
        "DRESSAGE_LOCAL_BWRAP_TOTAL_SERVERS", "DRESSAGE_LOCAL_BWRAP_BASE_PORT", "DRESSAGE_BLACKBOX_ACQUIRE_TIMEOUT_SEC"]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))')"

ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_ENTRY[@]}" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${COMM_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
