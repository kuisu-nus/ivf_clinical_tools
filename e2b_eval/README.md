# e2b_eval：在独立 E2B 沙箱中逐例跑三阶段任务

读取 `dataset/build_samples.py` 生成的 `samples.jsonl`，为每个病例创建一个独立的 E2B 沙箱，按 Dressage 样本里的三个阶段执行：

```
samples.jsonl
      │  每个病例一个独立沙箱，互不共享文件、服务或 OpenClaw 会话
      ▼
┌──────────────────────────── E2B sandbox ────────────────────────────┐
│                                                                     │
│ ① before_agent                                                      │
│    上传 before_agent_files                                          │
│    BlackboxServer /v1/rollout/register（启动沙箱内 OpenClaw）       │
│    执行 before_agent 命令：setup → 工具服务 → 注册 MCP → 工具说明   │
│    检查两个工具服务的 /health                                        │
│                                                                     │
│ ② agent                                                             │
│    /v1/sessions/<id>/messages → OpenClaw 回答，调用 claw-tools       │
│    保存回答、会话历史、dispatches.jsonl、OpenClaw 日志              │
│                                                                     │
│ ③ after_agent                                                       │
│    上传 after_agent_files（评分器和 metadata.json，标签此时才进沙箱）│
│    执行 after_agent 命令：gen_trace → stop_services → run_grader     │
│    解析 overall_score                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
outputs/e2b_eval/cases/<instance_id>/score.json + summary.json
```

本项目**不 import Dressage**，只依赖 `e2b` SDK 和 `httpx`。它直接使用沙箱内的 BlackboxServer HTTP 协议和 `samples.jsonl` 里已经写好的 `blackbox_execute_cmds`，因此样本格式与 Dressage 保持一致，但运行路径是独立的。

## 和仓库里其它目录的关系

| 目录 | 作用 |
|---|---|
| `dataset/build_samples.py` | 生成 `samples.jsonl` 和每例的 `prompt.md`、`metadata.json`、`service.env` |
| `openclaw_eval/` | 在本机（例如 H20）安装好的 OpenClaw 上跑同一套三阶段，不创建 E2B 沙箱 |
| `e2b_eval/`（本目录） | 每个病例一个 E2B 沙箱；沙箱内有 BlackboxServer、OpenClaw 和工具服务 |
| `task/`、`grader/` | 样本里 `before_agent_files` / `after_agent_files` 指向的任务文件和评分器 |

## 快速开始

```bash
cd ~/ivf_clinical_tools
pip install -r e2b_eval/requirements.txt

cp e2b_eval/.env.example e2b_eval/.env
# 编辑 e2b_eval/.env：填 E2B_* 和 AGENT_LLM_*

# 1. 只做样本校验和 local_path 映射，不创建沙箱
python3 -m e2b_eval --samples runs/openclaw_dev/samples.jsonl \
  --env-file e2b_eval/.env --dry-run --limit 2

# 2. 跑一个病例
python3 -m e2b_eval --samples runs/openclaw_dev/samples.jsonl \
  --env-file e2b_eval/.env --case 0974

# 3. 跑全部病例
python3 -m e2b_eval --samples runs/openclaw_dev/samples.jsonl \
  --env-file e2b_eval/.env
```

也可以直接用包装脚本；它会切到仓库根目录，并在 `e2b_eval/.env` 存在时自动加载：

```bash
bash e2b_eval/run.sh --samples runs/openclaw_dev/samples.jsonl --case 0974
```

默认输出到 `outputs/e2b_eval/`（已在 `.gitignore` 中）。已经生成 `score.json` 的病例会跳过；`--no-resume` 强制重跑。

常用参数：

| 参数 | 说明 |
|---|---|
| `--case ID` | 只跑匹配的病例；支持完整 `instance_id`、case key，或 `0974` 这样的后缀 |
| `--limit N` | 只跑前 N 个选中的病例 |
| `--workers N` | 同时跑 N 个沙箱（默认 1；一个病例一个沙箱） |
| `--out DIR` | 输出目录，默认 `outputs/e2b_eval` |
| `--dry-run` | 校验样本、命令和文件路径，不创建沙箱 |
| `--no-resume` | 已有 `score.json` 也重跑 |
| `--keep-sandbox` | 结束后不杀沙箱，便于进 dashboard 调试（沙箱到 timeout 才会消失） |
| `--task-root DIR` | 样本里的绝对 `local_path` 来自另一台机器时，用这个仓库根目录重映射 |
| `--path-map OLD=NEW` | 显式重映射 `local_path` 前缀，可重复 |
| `--template NAME` | 覆盖样本的 `metadata.sandbox_image` |
| `--sandbox-timeout SEC` | 覆盖样本的 `metadata.sandbox_timeout_sec` |
| `--agent-mode mock` | 不用 LLM，用确定性工具调用跑通 before/after，只验证流水线 |
| `--verbose` | debug 日志 |

## 环境变量

### E2B

| 变量 | 默认 | 说明 |
|---|---|---|
| `E2B_API_KEY` | — | 必填；也接受 `DRESSAGE_E2B_API_KEY` |
| `E2B_API_URL` | E2B SDK 默认 | 自托管 API，例如 `http://<host>:3000` |
| `E2B_SANDBOX_URL` | — | 自托管 client proxy，例如 `http://<host>:3002` |
| `E2B_TEMPLATE` | `e2b-dressage-claw-blackbox` | 默认模板名；样本的 `metadata.sandbox_image` 优先，`--template` 覆盖两者 |
| `E2B_SANDBOX_TIMEOUT_SEC` | 3600 | 默认沙箱寿命；样本的 `metadata.sandbox_timeout_sec` 优先，`--sandbox-timeout` 覆盖两者 |

**自托管 E2B 必须设置 `E2B_SANDBOX_URL`。** SDK 用它把 `commands.run`/文件操作路由到沙箱；只设置 `E2B_API_URL` 时，创建会成功，但命令会报 `TimeoutException: The sandbox was not found`。本项目在配置阶段就会给出明确错误。E2B Cloud 不设置这两个 URL。

自托管时，BlackboxServer 的访问方式是：请求发到 `E2B_SANDBOX_URL`，用 `Host: 31000-<sandbox_id>.e2b.app` 选择沙箱和端口。`before_agent.json` 里记录了实际 endpoint 和 Host。

### Agent LLM

沙箱里的 OpenClaw 通过 BlackboxServer 的 rollout proxy 调用一个 OpenAI 兼容接口。runner 本身不直接调用模型。

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容接口 |
| `AGENT_LLM_API_KEY` | — | `openclaw` 模式下必填；也接受 `OPENAI_API_KEY` |
| `AGENT_LLM_MODEL` | `deepseek-flash` | 传给 OpenClaw 的模型 id |
| `AGENT_LLM_PROVIDER_ID` | `deepseek` | OpenClaw provider id，可自定义 |
| `AGENT_LLM_CONTEXT_WINDOW` | 65536 | OpenClaw context window |
| `AGENT_LLM_MAX_TOKENS` | 4096 | 单次模型输出上限 |
| `AGENT_MAX_STEPS` | 40 | 最多工具调用步数；`0` 表示不限制 |
| `AGENT_REGISTER_TIMEOUT_SEC` | 300 | 等待 BlackboxServer 启动 OpenClaw 的时间 |
| `AGENT_TURN_TIMEOUT_SEC` | 900 | 单次 agent turn 的最长时间 |
| `E2B_BBS_RUNTIME_ROOT` | `/workspace_sandbox/blackbox_server_runtime` | 一般不用改 |

数据会离开沙箱到 `AGENT_LLM_BASE_URL`，请使用获批的模型服务和专用、可撤销的 key。API key 会写进沙箱内的 OpenClaw 配置；本地 `before_agent.json` 里的 register 响应已脱敏。带 `--keep-sandbox` 时沙箱会继续保留，注意其中含 key。

### 工具服务端口（可选）

runner 在 before_agent 之后用 `curl` 检查两个工具服务的 `/health`，默认 9201/9202。如果样本里的 `start_services` 命令改了端口，也要在 runner 环境里设置：

```bash
export IVF_SIMILAR_PORT=9201
export IVF_PREDICT_PORT=9202
```

## 样本格式

`samples.jsonl` 每行一个 JSON，本 runner 使用以下字段（其余字段原样保存）：

```json
{
  "prompt": "病例文本……",
  "label": "",
  "reward_fn": "claw_grader",
  "metadata": {
    "instance_id": "claw-ivf-staged-train-0974",
    "task_id": "ivf-clinical-tools_7bbd9a02dff1",
    "blackbox_type": "openclaw",
    "sandbox_image": "e2b-dressage-claw-blackbox",
    "sandbox_timeout_sec": 3600,
    "system_prompt_file": "/tmp/omni_task/runtime/system_prompt_prefix.txt",
    "before_agent_files": [
      {"path": "/tmp/omni_task/scripts/setup.sh", "local_path": "/path/to/repo/task/scripts/setup.sh"}
    ],
    "after_agent_files": [
      {"path": "/tmp/omni_grader/metadata.json", "local_path": "/path/to/cases/0974/metadata.json"}
    ],
    "blackbox_execute_cmds": {
      "before_agent": [
        {"name": "setup_task", "cmd": "bash /tmp/omni_task/scripts/setup.sh", "required": true, "timeout": 120}
      ],
      "after_agent": [
        {"name": "run_grader", "cmd": "bash /tmp/omni_grader/run.sh", "required": true, "timeout": 120}
      ]
    }
  }
}
```

约束：

- 只支持 `metadata.blackbox_type = openclaw`；这是 `e2b-dressage-claw-blackbox` 模板里的 BlackboxServer backend。
- `before_agent_files` 不能包含 `/tmp/omni_grader` 下的文件；评分器和标签只能在 agent 结束后上传。runner 会拒绝把 grader 文件放进 before 阶段。
- `system_prompt_file` 必须同时出现在 `before_agent_files` 里，因为 register 时 OpenClaw 要读它。
- `blackbox_execute_cmds` 的 `cmd` 必须是单行；BlackboxServer 的 `execute_cmd` 不接受换行。多步用 `;` 或 `&&`。
- 病例真实标签在 `after_agent_files` 的 `metadata.json` 中，不在 JSONL 的 `label` 字段里。不要把 `runs/` 或病例目录提交到仓库。

### local_path 重映射

样本可能是在另一台机器上生成的，`local_path` 是那台机器的绝对路径。解析顺序：

1. 路径本身存在；
2. `--path-map OLD=NEW`；
3. `--task-root` + `/task/`、`/grader/`、`/outputs/` 或 `/runs/` 后面的后缀；
4. `<samples.jsonl 所在目录>/cases/...`。

默认 `--task-root` 是当前仓库根目录。先跑 `--dry-run` 可以一次看到所有解析结果和缺失文件。

## 每个阶段做了什么

### ① before_agent

1. 创建沙箱（模板、timeout、instance_id/task_id metadata）。
2. 上传 `before_agent_files`（保留样本里的绝对目标路径）。
3. 等待 BlackboxServer `/health`。
4. `POST /v1/rollout/register`：绑定 session、启动沙箱内 OpenClaw；`system_prompt_file` 会被复制到 OpenClaw workspace 的 `AGENTS.md`，并追加“不要向用户提问”的策略。
5. 按顺序执行 `before_agent` 命令；任意 `required: true` 的命令返回非 0 就停止，不进入 agent。
   - `map_workspace`
   - `setup_task`
   - `start_services`（两个 IVF 工具服务）
   - `register_mcp`（把 `/tmp/omni_task/runtime/claw_tools_mcp.py` 注册为 OpenClaw MCP server）
   - `expose_tool_instructions`（把工具说明追加到 workspace 的 `AGENTS.md`）
6. 用 `curl` 检查 9201/9202 的 `/health`；不健康则不进入 agent。

### ② agent

1. 以 async 模式提交一个 user message（病例 prompt），然后轮询 turn 状态直到 `committed`/失败。
2. 取回完整 session history（含 system prompt、工具调用、工具结果）。
3. 从最后一条无 `tool_calls` 的 assistant 消息中取回答，写入 `answer.txt`。
4. 生成 `agent_messages.json`（`{"prompt", "response", "messages"}`）并写入沙箱 `/tmp/omni_grader/agent_messages.json`。
5. 读取 MCP 桥写下的 `/tmp/omni_task/trace/dispatches.jsonl`。这份 trace 由工具桥在真实 HTTP 调用处记录，不是模型自述。

agent 失败/超时/没有文本回答时，不会进入 after_agent，病例标记 failed；日志和已有产物仍会保存。

### ③ after_agent

1. 上传 `after_agent_files`：评分器文件、`metadata.json`（含真实标签）。**这些文件在 agent 运行期间不存在。**
2. 按顺序执行 `after_agent` 命令：
   - `patch_llm_judge`
   - `generate_trace`（用 agent_messages + dispatches 生成 `tool_trajectory.json`、`llm_messages.json`）
   - `stop_services`
   - `run_grader`（输出 `overall_score` JSON）
3. 解析 `run_grader` stdout（也兼容 `/tmp/omni_grader/grader_output.json`），写 `grader_output.json` 和 `score.json`。

## 产物

`outputs/e2b_eval/cases/<instance_id>/`：

| 文件 | 内容 |
|---|---|
| `case.json` | 最终状态、错误、用时、sandbox id、tool_calls |
| `plan.json` | 解析后的上传计划和命令 |
| `sample.json` | 原始样本行 |
| `before_agent.json` | endpoint、session、health、register 响应（已脱敏） |
| `before_files.json` | 上传到沙箱的 before 文件 |
| `before_agent_commands.json` / `before_agent.log` | 每条 before 命令的 rc、stdout、stderr、耗时 |
| `tools_health.json` | 两个工具服务的 /health |
| `agent_response.json` | BlackboxServer 的 turn 结果（输出、usage、turn id） |
| `history.json` | 完整会话历史（含 tool_calls/tool 结果） |
| `answer.txt` | 最终回答 |
| `agent_messages.json` | 写进沙箱评分目录的对话证据 |
| `dispatches.jsonl` | MCP 工具桥记录的真实工具调用（请求、响应、状态、耗时） |
| `agent_run.json` | 模式、模型、session/turn、用时、usage、MCP tool_calls |
| `openclaw_logs.txt` | OpenClaw gateway 的 stdout/stderr（截断后的尾部） |
| `after_files.json` | 上传到沙箱的评分器文件 |
| `after_agent_commands.json` / `after_agent.log` | 每条 after 命令的 rc、stdout、stderr、耗时 |
| `grader_output.json` | 评分器完整输出（overall_score + 各评分项） |
| `grader_tool_trajectory.json` | 评分器使用的工具轨迹 |
| `grader_llm_messages.json` | 评分器使用的对话消息 |
| `grader_agent_messages.json` | 沙箱内 agent_messages.json 的副本 |
| `service_logs.txt` | 两个工具服务的日志尾部 |
| `score.json` | 上述结果的精简汇总：score、prediction、truth、agent、grader |
| `error.txt` | 失败时的 Python traceback |

`outputs/e2b_eval/summary.json`：总病例数、成功/失败/跳过、平均分、答案分布、真值分布、不确定数、排除不确定后的准确率，以及每个病例的结果。

`agent_run.json` 里有两个工具调用计数：

- `usage.tool_calls`：BlackboxServer/OpenClaw 在模型层统计的 step 内 tool calls（本任务里通常是 0，因为任务工具是 MCP 工具）；
- `tool_calls`：`dispatches.jsonl` 的真实 MCP 调用数（本任务期望 3）。

以 `tool_calls` 为准。

## 两种 agent 模式

### `openclaw`（默认）

真实运行沙箱内的 OpenClaw。需要 `AGENT_LLM_API_KEY`。BlackboxServer 的 OpenClaw adapter 会：

- 启动 `openclaw gateway`；
- 起一个本地 rollout proxy，把模型请求转发到 `AGENT_LLM_BASE_URL`；
- 通过 `openclaw mcp set` 注册的任务 MCP server 暴露 `claw-tools__*` 工具；
- 配置变化后 OpenClaw 会热重载 MCP（日志里的 `[reload] config hot reload applied (mcp)`）。

### `mock`

```bash
python3 -m e2b_eval --samples runs/openclaw_dev/samples.jsonl --agent-mode mock --case 0974
```

不调用 LLM。它在沙箱里直接通过 MCP 桥调用和真实 agent 相同的两个工具，用返回的数字按固定模板生成回答，然后正常走 after_agent 和评分。它用来验证 E2B、文件上传、工具服务、MCP trace、评分器这条链路；分数不代表模型能力。mock 读取病例 `metadata.json` 的 `inputs`（age/amh/gn_days），不读取 `label`。

## 调试

- **先 `--dry-run`**：能看到样本解析、命令、每个本地文件是否存在。
- **`--keep-sandbox`**：保留沙箱；输出里会打印 sandbox id。E2B dashboard 里可以看进程、文件；用完手动 kill。
- **`--verbose`**：打开 debug 日志（httpx 的请求日志被压到 WARNING）。
- **`score.json` 已存在但想重跑**：加 `--no-resume`。默认会跳过已完成病例。
- **只跑一个病例**：`--case 0974`，避免浪费其它沙箱。

### 常见问题

| 现象 | 处理 |
|---|---|
| `TimeoutException: The sandbox was not found` | 自托管 E2B 没设 `E2B_SANDBOX_URL`。设置 client proxy 地址（如 `http://host:3002`） |
| `E2B_API_KEY is not set` | 设置 `E2B_API_KEY` 或 `DRESSAGE_E2B_API_KEY` |
| `AGENT_LLM_API_KEY is not set` | 设置 agent 模型 key，或先用 `--agent-mode mock` |
| `401 Authentication Fails ... Your api key ... is invalid` | `AGENT_LLM_API_KEY` 无效；最常见的是 `.env` 仍是模板占位符 `sk-...`。本次 0974 的完整变更与根因分析见 [CHANGES-AND-FAILURE-ANALYSIS.md](CHANGES-AND-FAILURE-ANALYSIS.md) |
| `register -> HTTP 409 conflict` | 一个 BlackboxServer 只允许一个 binding。不要复用沙箱；本项目每例新建沙箱。若手工调试，重启沙箱 |
| before_agent 某条命令失败 | 看 `before_agent.log`、`before_agent_commands.json`、`service_logs.txt` |
| 工具健康检查失败 | 看 `service_logs.txt`；确认 `setup_task` 安装了 numpy/fastapi/pydantic/uvicorn，端口是 9201/9202 |
| `tool_calls` 为 0 | 看 `before_agent.log` 的 `register_mcp` / `expose_tool_instructions`，以及 `openclaw_logs.txt`。正常应有 MCP 热重载日志 |
| `agent finished but returned no assistant text` | 看 `agent_response.json`、`history.json`、`openclaw_logs.txt`；可能是模型接口不可达或 OpenClaw 启动失败 |
| `openclaw_logs.txt` 里有 `Invalid config ... agents.defaults` | OpenClaw adapter 在尝试权限配置的兼容模式；只要后面出现 `gateway ready` 就是正常的启动重试 |
| 每条命令的 stderr 都有 `/bin/sh: ...: source: not found` | BlackboxServer 用 `/bin/sh -lc` 执行命令，模板 profile 里有 bash 专用的 `source`；不影响返回码和任务 |
| 样本文件找不到 | 用 `--task-root` 或 `--path-map OLD=NEW`；先 `--dry-run` |
| 想改工具端口 | 同时改样本 `start_services` 命令和 runner 的 `IVF_SIMILAR_PORT`/`IVF_PREDICT_PORT`，健康检查才一致 |

## 限制

- 一个病例只跑一个 OpenClaw turn；多轮追问/多 segment 训练不在本 runner 范围内。
- 只实现 `blackbox_type=openclaw`。
- 不依赖 Dressage，也不生成 Dressage 的训练轨迹；它面向评估和流水线验证。
- `--workers` 会并发创建多个沙箱；受 E2B 主机容量和模型接口限流约束。
- 任务工具服务写死在本仓库的 `task/services` 布局（`/tmp/omni_services`、`/tmp/omni_task`）；其它任务只要样本的 `blackbox_execute_cmds` 和文件布局兼容即可复用。

## 测试

```bash
python3 -m unittest discover -s e2b_eval/tests -v
```

单元测试不创建沙箱，覆盖样本校验、路径映射、命令失败、回答/评分解析、汇总、脱敏和 sandbox endpoint 的 Host header。端到端验证需要 E2B 和模型接口。
