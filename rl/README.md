# rl：在 H20 上用 Dressage 训练 IVF 结局任务（OpenClaw，GRPO）

按 Dressage 的 [Claw 实验流程](https://github.com/Accio-Lab/Dressage/blob/main/docs/blackbox-dressage-claw-experiment-zh.md) 训练本任务，改动只有两处：

| Claw 文档 | 本任务 |
|---|---|
| 数据：`prepare_dressage_claw_data.py` 从 Hugging Face 转换 441 个任务 | `dataset/build_samples.py` 从训练集生成，每个病例一条（同样的 jsonl 格式） |
| 沙箱：E2B 云沙箱（需要 key，且云上沙箱要能访问训练机） | 本机 bubblewrap 沙箱池（`local_bwrap`）。H20 没有 E2B key，也不能从外网访问 |
| 训练：4 节点 32 卡，CP=8（64k 上下文） | 单机 8 卡（`run_ivf_claw_h20.sh`），CP=4（32k）。本任务每例约 4 次模型调用，32k 足够 |

Rollout 和 reward 直接用 Dressage 的 Claw recipe：`dressage.recipes.dressage_claw.dispatch.generate`（分阶段写文件、执行 before/after 命令）和 `dressage.recipes.dressage_claw.reward`（读 `run_grader` 输出的 `overall_score`）。

> `run_ivf_claw_h20.sh` 由 Dressage 自带的单机脚本改写，还没有在 GPU 上运行过。先做第 4 步的冒烟测试。

## 0. 检查环境

```bash
python3 -c "import torch, sglang, ray; print('torch', torch.__version__, 'sglang', sglang.__version__, 'ray', ray.__version__)"
python3 -c "import fastapi, numpy; print(fastapi.__file__)"    # 不能在 ~/.local 下：沙箱里看不到用户目录
which bwrap openclaw hf
hostname -i                                                     # 不能是 127.x：沙箱要通过这个地址访问 Dressage Proxy
```
- `hostname -i` 是 127.x：运行时加 `PROXY_PUBLIC_HOST=<本机内网 IP>`
- H20 的容器由 Dressage 镜像建成，但镜像把 Dressage、Megatron-LM 装在 `/root` 下，非 root 账号读不到
  （`import dressage` 报 `PermissionError: /root/Dressage/...`）。按第 1 步使用 home 目录下的副本；
  训练脚本会把 `~/Dressage`、`~/Dressage/slime`、`~/Megatron-LM` 放进 `PYTHONPATH`，优先于 `/root` 下的安装。

## 1. Dressage、slime 和 Megatron-LM（home 目录下的副本）

```bash
cd ~/Dressage && git pull && git submodule update --init --recursive
git submodule status | head -1                                  # 行首不再是 "-"

# Megatron-LM：与镜像相同的版本和补丁（slime/docker/Dockerfile 的 MEGATRON_COMMIT 和 docker/patch/latest/megatron.patch）
mkdir ~/Megatron-LM && cd ~/Megatron-LM && git init -q && git remote add origin https://github.com/NVIDIA/Megatron-LM.git
git fetch --depth 1 origin 1dcf0dafa884ad52ffb243625717a3471643e087 && git checkout -q FETCH_HEAD    # 只拉这一个版本
git apply --3way ~/Dressage/slime/docker/patch/latest/megatron.patch && echo "patch ok"            # 21 个文件，均可干净应用
grep -R -n '^<<<<<<< ' . | head -3                              # 应该没有输出

export PYTHONPATH=$HOME/Megatron-LM:$HOME/Dressage:$HOME/Dressage/slime
python3 -c "import dressage, blackbox_server, slime, megatron.core; print('imports ok')"
```
slime 里目前只有 `qwen3.5-35B-A3B.sh`。Qwen3.6-35B-A3B 与 Qwen3.5-35B-A3B 结构相同（同一个模型类、同样的层数和维度），脚本会自动用这个文件；`TOKEN_BUILD_MODEL` 仍用 `qwen3_6`。

## 2. 模型：下载并转成 Megatron 格式

```bash
export MODEL_ROOT=~/models; mkdir -p $MODEL_ROOT
hf download Qwen/Qwen3.6-35B-A3B --local-dir $MODEL_ROOT/Qwen3.6-35B-A3B          # 约 70GB
cd ~/Dressage/slime && source scripts/models/qwen3.5-35B-A3B.sh                    # 与 Qwen3.6-35B-A3B 结构相同
PYTHONPATH=$HOME/Megatron-LM:$HOME/Dressage/slime torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" --hf-checkpoint $MODEL_ROOT/Qwen3.6-35B-A3B --save $MODEL_ROOT/Qwen3.6-35B-A3B_torch_dist
```
先调通流程时可以换小模型（例如 `Qwen3.5-4B`，设 `MODEL_NAME`、`MODEL_CONFIG=qwen3.5-4B.sh`、`TOKEN_BUILD_MODEL=qwen3_5`，并按需调小 `PERF_ARGS` 的并行度）。

## 3. 训练和验证样本

需要含 35 项指标的清洗后训练集（`train_clean.csv`，不要提交）。fold 0（约 226 例）只做验证，fold 1–4（约 906 例）用于训练：

```bash
cd ~/ivf_clinical_tools
python3 dataset/build_samples.py --cases ~/train_clean.csv --split train --fold 1 2 3 4 --out runs/rl_train --task-root "$(pwd)"
# 先用 64 条试跑：加 --limit 64（按哈希顺序取，各折都有）
python3 dataset/build_samples.py --cases ~/train_clean.csv --split train --fold 0 --out runs/rl_eval_fold0 --task-root "$(pwd)"
```
每条样本的工具都排除该病例所在的 1/5 数据（`service.env` → `start_services` 命令）。

## 4. 冒烟测试（2 步，每步 4 例 × 4 次采样）

```bash
cd ~/ivf_clinical_tools
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 RUN_NAME=ivf-smoke \
PROMPT_DATA=$(pwd)/runs/rl_train/samples.jsonl MODEL_ROOT=~/models \
bash rl/run_ivf_claw_h20.sh 2>&1 | tee runs/ivf-smoke.log
```
检查：
```bash
L=~/Dressage/log/traj_payload/ivf-smoke
grep -rhoE 'overall_score[^0-9]{1,6}[0-9.]+' $L | grep -oE '[0-9.]+$' | sort | uniq -c   # 每条轨迹都有分数，且不全是 0
grep -rho 'registered claw-tools[^"]*' $L | sort -u | head      # 每个沙箱用自己的端口（20000 + 沙箱编号）
grep -rhoE 'cases[^0-9]{1,6}9(05|06|07)|lr_35_indicators_fold[0-4]' $L | sort | uniq -c   # 工具排除了病例所在的折（905–907 例、fold 模型）
ls ~/Dressage/log/traj_err/ivf-smoke                            # 失败的轨迹及原因（见下方“排错”）
```

## 5. 正式训练

```bash
RUN_NAME=ivf-claw-v1 PROMPT_DATA=$(pwd)/runs/rl_train/samples.jsonl MODEL_ROOT=~/models \
nohup bash rl/run_ivf_claw_h20.sh > runs/ivf-claw-v1.log 2>&1 &
```
默认每步 8 例 × 8 次采样，共 100 步（约 800 个病例，接近训练集一遍），每 10 步保存到 `$MODEL_ROOT/Qwen3.6-35B-A3B_ivf_<RUN_NAME>/`。可调参数见脚本开头的注释。

要盯住的指标：
- 平均 reward 和组内方差：方差接近 0 时 GRPO 没有学习信号；
- 回答"不确定"的比例：它固定得 0.5，比例持续上升说明模型在走捷径；
- 每条轨迹的 segment 数：应该一直是 1。

## 6. 验证

把保存的 checkpoint 转回 Hugging Face 格式（slime 的 `tools/convert_torch_dist_to_hf.py`），用 SGLang 启动服务，把 OpenClaw 的模型指向它，再用 `openclaw_eval/run_all.sh runs/rl_eval_fold0` 在 fold 0 上评估。按平衡准确率和"不确定"比例挑 checkpoint。283 例封存测试集只在最后跑一次。

## 端口说明

bubblewrap 沙箱共用本机网络。如果每个沙箱都用 9201，后启动的沙箱会连到先启动的沙箱的服务，拿到排除了另一个病例的数据。所以 `task/scripts/ports.sh` 按沙箱编号（`DRESSAGE_BLACKBOX_SLOT_ID`）分配端口：20000 + 编号（所有工具都在这一个端口上）。`register_mcp.sh` 通过 `OMNI_TOOL_PORT_MAP` 告诉 MCP 工具桥实际端口。`start_services.sh` 还会检查回应健康检查的确实是刚启动的服务。只有数字编号（本机 bubblewrap）才换端口；E2B 中每个沙箱有自己的网络，Dressage 把编号设为 `e2b`，仍用 9201。

## 排错：before_agent 命令失败

Dressage 抛出的 `required execute_cmd failed: name=<命令> returncode=1` 只有命令名和退出码。命令的 stdout/stderr 在失败轨迹的 `error.json` 里：`$LOG_DIR/traj_err/<RUN_NAME>/<instance_id>/<session_id>/`，字段 `execute_cmds[]` → `cmd_result.stderr`（`LOG_DIR` 默认为运行训练的那份 Dressage 仓库下的 `log/`，例如 `/root/Dressage/log`）。成功轨迹的同一字段在 `traj_payload/` 下。按命令和报错统计失败原因：

```bash
python3 - ~/Dressage/log/traj_err/<RUN_NAME> <<'PY'
import collections, json, pathlib, sys
reasons = collections.Counter()
for f in pathlib.Path(sys.argv[1]).rglob("error.json"):
    stack = [json.loads(f.read_text())]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            result = x.get("cmd_result")
            if isinstance(result, dict) and result.get("returncode") not in (0, None):
                reasons[(x.get("name"), (result.get("stderr") or "").strip()[-300:])] += 1
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
for (name, err), n in reasons.most_common():
    print(f"{n} x {name}\n  {err}")
PY
```
`start_services` 失败时，脚本会把工具服务日志的最后 40 行写进 stderr。
