# openclaw_eval：在本机 OpenClaw 上跑 IVF 任务并评分

把 Dressage claw 样本的三个阶段搬到一台装了 OpenClaw 的机器上（例如 H20）逐个病例执行：

```
            一次性准备 prepare.sh：注册 MCP 工具 claw-tools，写入 AGENTS.md
                                   │
        ┌──────────────────────────┴───────────────────────────┐
        ▼                                                      │
 ① before_agent.sh ──服务就绪──▶ ② agent.sh ──回答 + 工具调用记录──▶ ③ after_agent.sh ──▶ score.json
   按病例设置启动工具服务          OpenClaw 回答病例              放入标签和评分器，打分
        ▲                                                      │
        └──────────────── run_all.sh：下一个病例 ◀─────────────────┘
                          全部完成后 summarize_eval.py → summary.json
```

| 本目录脚本 | 对应 Dressage 样本中的命令 |
|---|---|
| `prepare.sh` | `register_mcp`、`expose_tool_instructions`（以及 BlackboxServer 把 system prompt 写成 AGENTS.md） |
| `before_agent.sh` | `start_services`（`setup_task` 在本机已具备，`map_workspace` 本任务不需要） |
| `agent.sh` | Dressage 通过 BlackboxServer 调用 OpenClaw；这里直接 `openclaw agent` |
| `after_agent.sh` | 写入 `after_agent_files`（评分器、metadata.json）+ `generate_trace` + `run_grader` |

工具代码、MCP 工具桥、system prompt、评分器都和 Dressage 样本用的是同一份文件。

## 快速开始（H20）

```bash
cd ~/ivf_clinical_tools && git pull origin wzy

# 生成病例（每例一个目录：prompt.md、metadata.json、service.env）
python3 dataset/build_samples.py --cases <病例CSV> --split train --fold 0 --out runs/openclaw_dev --task-root "$(pwd)"

bash openclaw_eval/prepare.sh                       # 一次：注入工具、写入 AGENTS.md（原文件自动备份）
bash openclaw_eval/run_all.sh runs/openclaw_dev     # 逐个病例跑三个阶段并汇总
bash openclaw_eval/restore.sh                       # 用完恢复：移除工具、还原 AGENTS.md / BOOTSTRAP.md
```

只跑一个病例、或逐阶段查看时：

```bash
C=runs/openclaw_dev/cases/0062; R=runs/openclaw_dev/eval_openclaw/cases/0062
bash openclaw_eval/before_agent.sh $C $R
bash openclaw_eval/agent.sh        $C $R
bash openclaw_eval/after_agent.sh  $C $R
```

`run_all.sh` 中断后重新运行即可，已有 `score.json` 的病例会跳过。

## 每个阶段的输入和输出

病例目录 `<samples>/cases/<id>/`（`build_samples.py` 生成，**含结局标签，不要提交**）：

| 文件 | 内容 | 谁用 |
|---|---|---|
| `prompt.md` | 病例文本，分初诊 / 促排卵与取卵 / 胚胎三段，不含编号和结局 | agent |
| `service.env` | 工具服务设置。训练病例：`IVF_EXCLUDE_CASE`（排除该例所在的 1/5 数据）、`IVF_MODEL_NAME`（对应 fold 的模型）；测试病例为空 | before_agent |
| `metadata.json` | 真实结局 `label`，以及年龄、AMH、Gn 天数（检查工具参数用） | after_agent |

结果目录 `<eval>/cases/<id>/`：

| 阶段 | 文件 | 内容 |
|---|---|---|
| ① before_agent | `before_agent.json` | 本例的服务设置，以及两个服务的健康状态（数据范围、病例数、模型） |
| | `before_agent.log` | 启动日志 |
| ② agent | `openclaw_raw.txt` | `openclaw agent` 的原始输出 |
| | `answer.txt` | 去掉 `│ ◇` 进度行后的回答（【初诊】【促排】【胚胎】+ 最后一行结论） |
| | `dispatches.jsonl` | 每次工具调用一行：工具名、参数、返回、状态码、耗时（MCP 工具桥记录） |
| | `agent_run.json` | 会话 id、用时、OpenClaw 退出码、工具调用次数 |
| | `openclaw_stderr.txt` | OpenClaw 的错误输出 |
| ③ after_agent | `grader_mount/agent_messages.json` | 病例文本 + 回答（与 Dressage 写入沙箱的格式相同） |
| | `grader_mount/tool_trajectory.json` | 由 dispatches.jsonl 转换的工具调用轨迹（评分器输入） |
| | `grader_mount/llm_messages.json` | 对话消息（评分器输入） |
| | `grader_mount/metadata.json` | 标签，agent 结束后才放入 |
| | `grader_output.json` | 评分器输出：`overall_score` 和各项得分、原因 |
| | `score.json` | `grader_output.json` + `agent_run.json`，供 `summarize_eval.py` 汇总 |

汇总 `<eval>/summary.json`：回答分布、准确率、敏感度、特异度、平衡准确率、"不确定"比例、平均分及各项平均分、平均字数。

### 评分（`grader/grader.py`，确定性，无 LLM judge）

| 项 | 权重 | 规则 |
|---|---|---|
| outcome | 0.60 | 结论与真实结局比较，按类别频率平衡：答对妊娠 1.0、答对未妊娠 0.823、答错 0 或 0.177、不确定 0.5（总是答同一个结论的期望得分都是 0.5） |
| tool_use | 0.15 | 初诊预测（年龄、AMH）、促排预测（加 Gn 天数）、相似病例检索各一次，参数等于病例值 |
| faithfulness | 0.10 | 回答中的百分数和例数都来自工具返回 |
| format | 0.05 | 三个阶段按顺序出现，结论是最后一行 |
| concise | 0.10 | 300 字以内满分，600 字为 0 |
| 门槛 | — | 给出治疗或用药建议，整题 0 分 |

## 示例：`examples/`

`examples/cases/demo/` 是一个虚构病例（38 岁，AMH 0.9，Gn 10 天，与 `task/prompt.md` 相同），按 `build_samples.py` 的格式给出三个输入文件；其中的 `label` 只为演示结局评分。
`examples/output/demo/` 是它的一次完整运行结果，文件与上表一一对应。

这份示例在开发环境中生成，那里没有 OpenClaw，agent 阶段由一个调用同一 MCP 工具桥的替身完成：工具服务、工具调用记录、轨迹转换和评分都是真实运行的结果，`answer.txt` 的文字是按输出格式填入工具数值的模板，不是模型的回答（所以 `agent_run.json` 的用时为 0）。真实的 OpenClaw 运行结果见 H20 上的 `<eval>/cases/<id>/`。

可以直接对示例跑一遍流程：`bash openclaw_eval/run_all.sh openclaw_eval/examples /tmp/demo_eval`。

## 设置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `IVF_SIMILAR_PORT` / `IVF_PREDICT_PORT` | 9301 / 9302 | 工具服务端口。改了之后要重新运行 `prepare.sh` |
| `OC_WORKSPACE` | `~/.openclaw/workspace` | 使用哪个 OpenClaw 工作区的 AGENTS.md |
| `OC_EVAL_WORK` | `outputs/openclaw` | 工具配置、调用记录、服务日志、备份的位置 |
| `OC_LOCAL` | 0 | 设为 1 时用 `openclaw agent --local`（不经过 Gateway） |
| `OC_TIMEOUT` | 600 | 每个病例 agent 的超时秒数 |
| `OPENCLAW_BIN` / `PYTHON` | `openclaw` / `python3` | 可执行文件 |

## 注意

- **端口**：H20 的 9201/9202 上有一套别人启动的、用全部数据的服务。用它会让相似病例检索查到病例本身，所以这里默认用 9301/9302。`before_agent.sh` 会检查回答健康检查的确实是刚启动的服务、模型和数据范围符合本例设置，不符合就停下并说明原因。
- **AGENTS.md**：`prepare.sh` 会覆盖工作区的 AGENTS.md，并把 BOOTSTRAP.md 移走（否则 OpenClaw 可能先进行首次见面的引导对话）；原文件在 `outputs/openclaw/backup/`，`restore.sh` 还原。
- **数据**：病例目录和结果目录都含真实结局标签，放在 `runs/`、`outputs/` 下（已在 .gitignore 中），不要提交。
