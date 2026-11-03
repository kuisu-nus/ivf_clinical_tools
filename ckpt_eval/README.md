# ckpt_eval：评估 Dressage 训练出的 checkpoint

把训练保存的 checkpoint 转成 Hugging Face 格式，用 SGLang 部署，接到 OpenClaw 上，用同一批样本评估多个模型并对比。
2026-11 在 H20 上用它评估了 Qwen3-4B 训练前、第 100 步、第 400 步三个模型（283 例测试集，三个模型并行，约 1.5 小时）。

| 步骤 | 脚本 | 作用 |
|---|---|---|
| 1. 模型转换 | `convert.sh` | Megatron torch_dist（`iter_XXXXXXX`）→ Hugging Face，并检查权重名与原模型完全一致（`check_weights.py`） |
| 部署 | `serve.sh` | 用 SGLang 在指定 GPU 和端口后台启动模型，等到可用为止 |
| 2a. OpenClaw（评估用） | `oc_home.sh` | 每个模型一份独立的 OpenClaw 目录：指向该模型、精简工作区文件、注册两个 IVF 工具；不改原来的 `~/.openclaw` |
| 2b. OpenClaw（给人体验） | `oc_deploy.sh` / `oc_restore.sh` | 把主 OpenClaw 切到某个模型，换上任务的 AGENTS.md 和工具；先备份，可一键还原 |
| 3. 测试数据评估 | `run_eval.sh` → `status.sh` → `compare.py` | 多个模型同时跑完一批样本；看进度；出对比表，可导出全部回答 |

## 前提

- 已按 `rl/README.md` 准备好 `~/Dressage`（含 slime）和 `~/Megatron-LM`，或用 `DRESSAGE_ROOT`、`MEGATRON_ROOT` 指定。
- 本机装好 OpenClaw，`~/.openclaw/openclaw.json` 可用（脚本以它为模板）。
- 服务器上 `HOME` 被设成 `/root` 时，先 `export HOME=/home/<用户名>`。
- 评估样本用 `dataset/build_samples.py` 生成，例如 283 例测试集：
  `python3 dataset/build_samples.py --cases ~/ivf_data/xgb_test_283.csv --split test --out runs/test283 --task-root "$(pwd)"`

## 完整流程（以训练前、第 100 步、第 400 步为例）

```bash
M=~/Dressage/models; CK=$M/Qwen3-4B-Instruct-2507_slime

# 1. 转换（CPU，每个约几分钟；最后一行 missing [] extra [] 表示权重齐全）
bash ckpt_eval/convert.sh $CK/iter_0000099 $M/Qwen3-4B-ivf-step100-hf $M/Qwen3-4B-Instruct-2507
bash ckpt_eval/convert.sh $CK/iter_0000399 $M/Qwen3-4B-ivf-step400-hf $M/Qwen3-4B-Instruct-2507

# 部署：每个模型一张卡、一个端口（先用 nvidia-smi 看哪些卡空闲）
bash ckpt_eval/serve.sh $M/Qwen3-4B-Instruct-2507  step0   30000 1
bash ckpt_eval/serve.sh $M/Qwen3-4B-ivf-step100-hf step100 30001 2
bash ckpt_eval/serve.sh $M/Qwen3-4B-ivf-step400-hf step400 30002 3

# 2a. 每个模型一份 OpenClaw（名称 = SGLang 的 served name；工具端口每份不同）
bash ckpt_eval/oc_home.sh step0   30000 9311
bash ckpt_eval/oc_home.sh step100 30001 9321
bash ckpt_eval/oc_home.sh step400 30002 9331

# 3. 评估：后台并行，可以断开终端；中断后重跑会跳过已完成的病例
bash ckpt_eval/run_eval.sh runs/test283 step0 step100 step400
bash ckpt_eval/status.sh test283 step0 step100 step400          # 随时查看进度
python3 ckpt_eval/compare.py ~/oc_eval/{step0,step100,step400}/test283 --export ~/oc_eval/responses_test283.jsonl
```

`compare.py` 输出一张 Markdown 表：结论分布、平衡准确率、敏感度、特异度、答"妊娠"/"未妊娠"时各对了多少、平均 reward 和各分项、平均工具调用次数，以及各模型逐例结论的组合。`--export` 把每个模型每个病例的回答、各项得分和评分理由写成 JSONL。

每个病例的原始记录在 `~/oc_eval/<模型>/<样本集>/cases/<病例>/`：`answer.txt`（回答）、`dispatches.jsonl`（工具调用和返回）、`grader_output.json`（每项得分和理由）、`score.json`、`agent_run.json`（会话编号、用时）。

## 给 mentor 体验某个模型（2b）

```bash
HOST=0.0.0.0 bash ckpt_eval/serve.sh ~/Dressage/models/Qwen3-4B-ivf-step100-hf ivf-step100 30001 1   # 0.0.0.0：其他机器也能连
bash ckpt_eval/oc_deploy.sh 30001 ivf-step100       # 打印备份目录
openclaw agent --session-id demo --message "病例或问题"   # 或 openclaw tui
bash ckpt_eval/oc_restore.sh <备份目录>               # 体验完还原
```

已在运行的 OpenClaw 网关会自动读到新配置，不需要重启。

## 注意

- 新加的模型提供方用占位 API key：OpenClaw 把真 key 存在自己的认证库里，新提供方在库里没有记录，不填会报 `No API key found`；本地 SGLang 不检查 key。
- 生成样本后再改动了 `task/` 下的文件（比如工作区 md），要重新生成样本：样本里写死了要上传的文件清单。
- `runs/`、`~/oc_eval/` 和导出的 JSONL 都含病例数据和结局，不要提交。283 例封存测试集只在最终评估时用；比较 prompt 或奖励的改动请用验证集（`--split train --fold 0`）。
- 关模型：`pkill -f "served-model-name <名称>"`。关某份 OpenClaw 的工具服务：`(source ~/oc_eval/<名称>/env.sh && bash task/scripts/stop_services.sh)`。
