# ivf_clinical_tools 变更说明与失败分析

- 日期：2026-10-28
- 改动范围：`e2b_eval/`（新增）、`.gitignore`、根 `README.md`
- 分析对象：2026-10-28 02:21 病例 `claw-ivf-staged-train-0974` 的 E2B + OpenClaw 运行失败

## 摘要

上一轮工作在 `ivf_clinical_tools` 中新增了独立的 E2B 三阶段评估项目 `e2b_eval/`，并修改了 `.gitignore` 和根 `README.md`。**MCP 服务本身的代码没有改动**：`task/runtime/claw_tools_mcp.py`、`task/services/*`、`task/scripts/*` 在本轮工作树中均无 diff。变化只发生在每次沙箱运行时：由 `before_agent` 启动两个工具服务、注册 MCP server、把工具说明写入 `AGENTS.md`，然后做健康检查。

2026-10-28 02:21 的 0974 用例失败，**直接原因不是 MCP，也不是 E2B/沙箱**，而是：

> `e2b_eval/.env` 第 11 行的 `AGENT_LLM_API_KEY` 仍然是 `.env.example` 里的占位符 `sk-...`。

这个值被原样传给沙箱内的 OpenClaw，OpenClaw 调 DeepSeek 时返回 `401 Authentication Fails`。同一次运行的 MCP 注册（`register_mcp`）和两个工具服务的 `/health` 都成功，说明工具侧链路是通的。

修复方式：把 `e2b_eval/.env` 里的 `AGENT_LLM_API_KEY` 换成真实可用的 key，按第 5 节做一次接口预检，再用 `--case 0974 --no-resume` 重跑。建议后续在 `config.py` 增加占位符校验，避免再浪费一次约 50 秒的沙箱创建。

---

## 1. 背景与目标

`e2b_eval/` 的目标是：读取 `dataset/build_samples.py` 生成的 `samples.jsonl`，为每个病例创建一个**独立 E2B 沙箱**，在沙箱内按 Dressage claw 样本的三阶段执行：

1. **before_agent**：上传任务文件、启动 OpenClaw/BlackboxServer、启动两个工具服务、注册 MCP、暴露工具说明、检查服务健康。
2. **agent**：通过 BlackboxServer 把病例 prompt 交给沙箱内的 OpenClaw，模型按需调用 `claw-tools__*` MCP 工具，保存回答和工具调用轨迹。
3. **after_agent**：**只有 agent 成功后才上传评分器和标签**，生成轨迹并运行确定性评分器，写出 `score.json`。

它不 import Dressage，只依赖 `e2b` SDK 和 `httpx`；样本格式与 Dressage 保持一致，但运行路径独立。每个病例一个沙箱，互不共享文件、服务或 OpenClaw 会话。

---

## 2. 变更清单

### 2.1 仓库改动（`git status` / `git diff`）

| 文件 | 类型 | 内容 |
|---|---|---|
| `e2b_eval/` | 新增（未跟踪） | 独立 E2B 三阶段评估项目，见下表 |
| `.gitignore` | 修改 | 新增 `runs/`、`.env`，避免病例数据、运行产物和密钥被提交 |
| `README.md` | 修改 | 增加“在 E2B 沙箱上评估”小节和使用示例 |

`e2b_eval/` 文件职责：

| 文件 | 职责 |
|---|---|
| `__init__.py` / `__main__.py` | 包入口、版本号、`python3 -m e2b_eval` 入口 |
| `cli.py` | 命令行参数、`--env-file` 加载、病例筛选、`--dry-run`、并发数 |
| `config.py` | `E2BConfig`、`AgentLLMConfig` 的环境变量解析与基本校验 |
| `samples.py` | 解析 `samples.jsonl`，校验并映射 `before_agent_files` / `after_agent_files`，提取命令 |
| `sandbox.py` | 创建/销毁 E2B 沙箱、上传/读取文件、执行命令；为自托管 client proxy 设置 `Host` 头 |
| `blackbox.py` | BlackboxServer HTTP 客户端：健康检查、`/v1/rollout/register`、提交 turn、轮询、脱敏 |
| `runner.py` | 三阶段编排、before 命令执行、MCP 注册、工具健康检查、读取 `dispatches.jsonl`、调用评分器、汇总 |
| `util.py` | 环境文件加载（默认不覆盖已有变量）、JSON/文本读写、密钥脱敏 |
| `tests/` | 46 个单元测试：样本校验、路径映射、命令失败、回答/评分解析、汇总、脱敏、Host header |
| `run.sh` | 包装脚本：切到仓库根目录，存在 `.env` 时自动传 `--env-file` |
| `.env.example` | E2B 与 `AGENT_LLM_*` 配置模板（含占位符 key） |
| `README.md` | 用法、环境变量、产物、常见问题、限制 |

### 2.2 MCP 服务侧：代码未改，只有运行期动作

**如果你说的“MCP 服务”指 `task/runtime/claw_tools_mcp.py` 和 `task/services/*`，那么本轮没有修改它们。** 本次新增的是围绕这些既有服务运行的 E2B runner；服务代码、工具 spec、模型文件都沿用仓库原样。

每次 `before_agent` 会按样本里的命令顺序执行 5 步：

```text
map_workspace
  → setup_task        # 安装 numpy/fastapi/pydantic/uvicorn
  → start_services    # 启动 ivf_similar:9201、ivf_predict:9202
  → register_mcp      # openclaw mcp set claw-tools ...
  → expose_tool_instructions  # 追加 2 个工具到 AGENTS.md
```

其中 `register_mcp` 的实际内容是：

```bash
OC_HOME=$(ls -d "${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}"/bbs-*/home | head -1)
HOME="$OC_HOME" openclaw mcp set claw-tools \
  '{"command":"python3","args":["/tmp/omni_task/runtime/claw_tools_mcp.py"],
    "env":{
      "OMNI_TOOL_SPECS_DIR":"/tmp/omni_task/tools",
      "OMNI_TRACE_DIR":"/tmp/omni_task/trace",
      "NO_PROXY":"127.0.0.1,localhost,0.0.0.0,::1",
      "no_proxy":"127.0.0.1,localhost,0.0.0.0,::1"
    },
    "cwd":"/tmp/omni_task"}'
```

MCP 工具桥 `claw_tools_mcp.py` 的职责：

- `tools/list`：读取 `OMNI_TOOL_SPECS_DIR` 下的 `*.json`（本任务为 `ivf_predict_outcome.json`、`ivf_similar_patients.json`）；
- `tools/call`：把调用转发到 spec 里的 HTTP 后端（`127.0.0.1:9201/9202`）；
- 每次调用追加到 `OMNI_TRACE_DIR/dispatches.jsonl`；
- `--append-agents-md` 把工具说明追加到 OpenClaw 工作区的 `AGENTS.md`。

agent 阶段的工具调用轨迹由 runner 从沙箱内 `/tmp/omni_task/trace/dispatches.jsonl` 读回（`e2b_eval/runner.py:41`），保存为 `dispatches.jsonl`，并转换为评分器需要的工具轨迹。

### 2.3 安全约束

- **标签隔离**：`after_agent_files`（评分器 + `metadata.json` 里的真实结局）只在 agent 成功后才上传；agent 运行时沙箱里没有标签。
- **路径校验**：`before_agent_files` 中如果出现 `/tmp/omni_grader` 路径会被拒绝，防止提前把评分证据放进沙箱。
- **脱敏**：`before_agent.json` 里的 register 响应会把 `api_key` 替换为 `<redacted>`；日志中不写真实 key。
- **忽略**：`runs/`、`.env`、`outputs/` 都在 `.gitignore` 中。

---

## 3. 上一轮验证基线（2026-10-27）

上一轮完成时做过以下验证：

| 验证 | 结果 |
|---|---|
| 单元测试 | 46 passed |
| Mock 端到端（0974，无 LLM） | score 1.000，3 次 MCP 工具调用 |
| 真实 OpenClaw（0974，`deepseek-flash`） | score **0.975**，`tool_calls=3`，agent 约 10.0s，整例 58.9s，预测“妊娠” / 真值“妊娠” |

真实运行得 0.975 而不是 1.0，原因是回答里引用了 2 个并非工具返回的数字，`faithfulness=0.75`；其余项（outcome、tool_use、format、concise、no_treatment_advice）都是 1.0。这是模型行为，不是流水线错误。

> 说明：这次成功运行使用的 key 是在启动命令里通过 shell `export` 的，不经过 `.env`。该次产物目录后来被 2026-10-28 的失败运行覆盖，分数数据来自当时的会话记录；失败运行的证据仍完整保留在 `outputs/e2b_eval/`。

---

## 4. 本次失败分析（2026-10-28 02:21）

### 4.1 现象与时间线

运行命令（从 bash 历史与会话记录还原）：

```bash
bash e2b_eval/run.sh --samples runs/openclaw_dev/samples.jsonl --case 0974 --no-resume
```

| 时间（UTC） | 事件 |
|---|---|
| 02:08:07 | `e2b_eval/.env` 被创建/编辑（mtime） |
| 02:21:06.685 | 病例开始 |
| 02:21:07.465 | BlackboxServer `register` 返回 `status=ready`，OpenClaw 启动 |
| 02:21:07 之后 | `map_workspace` / `setup_task` / `start_services` 全部 rc=0；两个工具服务健康检查通过 |
| 02:21:56 前 | `register_mcp` rc=0：`Saved MCP server "claw-tools"`；`expose_tool_instructions` 追加 2 个工具到 `AGENTS.md` |
| 02:21:56.033 | OpenClaw 检测到 MCP 配置变化并热重载：`config hot reload applied (mcp)` |
| 02:21:59.752 | 第一次模型调用失败：`401 Authentication Fails, Your api key: ****-... is invalid` |
| 02:22:00.103 | runner 写 `error.txt`、`case.json`，随后写 `outputs/e2b_eval/summary.json`；病例标记 failed |

结果：

- `case.json`：`status=failed`，`duration_ms=53422`，`agent_mode=openclaw`，`tool_calls=null`，`answer_chars=0`
- `summary.json`：`cases=1, ok=0, failed=1, skipped=0`
- 失败阶段是 **agent**，不是 before_agent：

```text
e2b_eval.blackbox.BlackboxError: agent turn ... unknown:
Backend request failed: openclaw /v1/chat/completions returned 401:
{"error":{"message":"401 Authentication Fails, Your api key: ****-... is invalid", ...}}
```

### 4.2 直接原因与证据链

**根因：`e2b_eval/.env` 里的 `AGENT_LLM_API_KEY` 是模板占位符。**

| 证据 | 内容 |
|---|---|
| `e2b_eval/.env:11` | `export AGENT_LLM_API_KEY="sk-..."` |
| `e2b_eval/.env.example:11` | 同样的 `export AGENT_LLM_API_KEY="sk-..."`；两值完全一致（长度 6，不是真实 key） |
| `.env` 与 `.env.example` 的 diff | E2B 地址和 E2B key 被填过；`AGENT_LLM_API_KEY` 一行仍是模板值（未改） |
| `e2b_eval/config.py:167-181` | 只检查 `resolved_key` 是否为空（`require_key and not resolved_key`），不检查格式或占位符 |
| `e2b_eval/blackbox.py:92-101` | 把 `agent.api_key` 原样放进 register 的 `backend_options.api_key` |
| `openclaw_logs.txt:19-20` | `Config last-known-good promotion skipped: redacted secret placeholder at models.providers.deepseek.apiKey` |
| `openclaw_logs.txt:21-26` | 401 `Authentication Fails`；`reason=auth`；没有可 fallback 的候选模型 |
| `error.txt:13` | runner 最终抛出的 `BlackboxError`（401） |

也就是说：

```text
.env 里的 sk-...
  → AgentLLMConfig.from_env() 只判非空，接受
  → BlackboxClient.register() 放进 backend_options.api_key
  → BlackboxServer 写入沙箱内 OpenClaw 配置
  → OpenClaw 调 https://api.deepseek.com/v1/chat/completions
  → DeepSeek 返回 401（把 key 掩码成 ****-...）
```

DeepSeek 的报错把 key 掩码了；但 `.env` 里的实际值就是 `sk-...`，与 `****-...` 的掩码形式吻合，且 OpenClaw 日志明确写了 “redacted secret placeholder”。

### 4.3 为什么上一轮成功、这一次失败

- **上一轮（10-27 09:30）**：真实运行是在 shell 里先 `export AGENT_LLM_API_KEY=sk-...（真实值）`，再直接执行 `python3 -m e2b_eval ...`；`--env-file` 不是必需路径，所以没有用到模板占位符。
- **本次（10-28 02:08）**：从 `.env.example` 复制出 `.env`，只替换了 `E2B_API_URL` / `E2B_SANDBOX_URL`（以及填了 E2B key），**没有替换 `AGENT_LLM_API_KEY`**。
- **本次（10-28 02:21）**：使用 `run.sh` 启动；`run.sh:11` 在参数前插入 `--env-file "$HERE/.env"`，`cli.py:146-147` 加载该文件，`util.py:87-103` 的加载逻辑是 `override=False`（不覆盖已存在的环境变量）。该终端没有预先 export 真实 key，因此 `.env` 中的 `sk-...` 生效。

> 注意：`run.sh` 不覆盖已经 export 的变量。临时办法是在 shell 里 export 真实 key 后再跑；但长期应直接修正 `.env`，并在代码里拒绝明显占位符。

### 4.4 排除项（这些不是失败原因）

| 疑点 | 证据 | 结论 |
|---|---|---|
| MCP 注册失败 | `before_agent.log:33-44`：`register_mcp` rc=0，`Saved MCP server "claw-tools"`，`append_agents_md: appended 2 tool(s)` | 正常 |
| 工具服务不可用 | `tools_health.json`：两个服务均返回 `status=ok`；`ivf_similar` cases=905，`ivf_predict` 模型 `lr_35_indicators_fold0.json` | 正常 |
| E2B / BlackboxServer | `before_agent.json`：`health.healthy=true`，`register.status=ready`，session 已绑定 | 正常 |
| 网络不通 | 401 是 DeepSeek 返回的 JSON 鉴权错误，不是连接超时/拒绝 | key 无效，不是网络 |
| OpenClaw gateway 启动期 `Invalid config ... agents.defaults` | 本次失败运行和 10-27 成功运行都出现过；之后都出现 `gateway ready` | 已知启动重试，非致命 |
| `/bin/sh: ... source: not found` | 每条 before 命令的 stderr 都有，但 rc=0 | `/bin/sh -lc` 执行含 bash 专用 `source` 的 profile，非致命 |
| OpenClaw 没有加载 MCP | `openclaw_logs.txt` 有 `config hot reload applied (mcp)` | 已加载；本次根本没有走到工具调用，401 发生在第一次模型请求 |

### 4.5 次要问题与风险

1. **占位符不早失败**：`config.py` 只判空，直到创建沙箱、注册 OpenClaw、发出第一次模型请求才拿到 401，浪费约 50 秒和一次沙箱。
2. **占位符容易漏改**：`.env.example` 的 `sk-...` 与真实 key 前缀相同，复制后只改 E2B 地址很容易漏掉。
3. **key 的落点**：key 会被写进沙箱内 OpenClaw 配置；如果使用 `--keep-sandbox`，沙箱在超时前一直保留，需注意清理，不要把带 key 的沙箱/日志外传。
4. **失败用例的重跑**：失败用例不会生成 `score.json`，默认 resume 会重跑；如果目录里已有旧的 `score.json`，需要 `--no-resume` 才会覆盖。

---

## 5. 修复步骤与验证

### 5.1 填入真实 key

编辑 `e2b_eval/.env` 第 11 行（不要把 key 打到终端或提交到仓库）：

```bash
nano e2b_eval/.env
# 把 export AGENT_LLM_API_KEY="sk-..." 改成真实、获批的 key
```

检查是否仍是模板占位符（不输出 key）：

```bash
grep -q '^export AGENT_LLM_API_KEY="sk-\.\.\."$' e2b_eval/.env \
  && echo "仍是占位符，请修改" || echo "已改过"
```

### 5.2 模型接口预检

加载 `.env` 后发一个最小请求，只看 HTTP 状态码：

```bash
set -a; source e2b_eval/.env; set +a

curl -sS -o /tmp/ivf_llm_preflight.json -w 'HTTP %{http_code}\n' \
  "$AGENT_LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $AGENT_LLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$AGENT_LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}"
```

- `HTTP 200`：key 可用，可以重跑。
- `HTTP 401`：key 仍无效，检查是否复制完整、是否过期/额度不足、是否与 `AGENT_LLM_BASE_URL` 匹配。

### 5.3 重跑 0974

```bash
bash e2b_eval/run.sh --samples runs/openclaw_dev/samples.jsonl --case 0974 --no-resume
```

预期：

- `before_agent.log` 的 5 条命令全部 rc=0；`tools_health.json` 两个服务 200；
- `agent_run.json` 的 `tool_calls` 为 3（真实 MCP 调用数）；
- 生成 `answer.txt`、`grader_output.json`、`score.json`；
- `summary.json` 中 `ok=1, failed=0`；
- 分数由模型回答决定，不保证与上一轮 0.975 相同。

### 5.4 如果仍然 401

1. 确认 `.env` 的 key 是最新且完整的（不要只复制了前缀）；
2. 确认该 key 对 `AGENT_LLM_BASE_URL`/`AGENT_LLM_MODEL` 有权限、账户有余额；
3. 看 `outputs/e2b_eval/cases/<case>/openclaw_logs.txt` 的 `reason=auth` / `candidate_failed`；
4. 若 key 曾在日志、截图或聊天中暴露，直接轮换（revoke + 新建）。

### 5.5 建议的加固（本次只写入文档，未改代码）

在 `AgentLLMConfig.from_env()` 里拒绝明显的占位符，尽早抛 `ConfigError`：

```python
# e2b_eval/config.py，解析出 resolved_key 之后
if mode != "mock" and resolved_key:
    stripped = resolved_key.strip()
    is_placeholder = (
        stripped in {"sk-...", "sk-…"}
        or (stripped.startswith("sk-") and set(stripped[3:]) <= {".", "…"})
    )
    if is_placeholder:
        raise ConfigError(
            "AGENT_LLM_API_KEY is still the .env.example placeholder; "
            "set a real key or use --agent-mode mock"
        )
```

配套在 `e2b_eval/tests/test_config.py` 增加用例：`sk-...` 必须报错、正常长度的 key 必须通过。

> 不要对 key 长度做硬性下限（例如直接要求 ≥20 字符），因为本地 OpenAI 兼容服务可能使用 `EMPTY` 这类短 token；只拒绝明确的省略号占位符最稳妥。

可选的进一步改进：在创建沙箱之前用 1-token 请求做一次模型接口预检。这样 401 会在本地立即报错，不会先花约 50 秒创建沙箱。

---

## 6. 附录

### 6.1 证据文件

| 路径 | 用途 |
|---|---|
| `outputs/e2b_eval/cases/claw-ivf-staged-train-0974/error.txt` | 失败 traceback，最后一行是 401 |
| `.../case.json` | 最终状态、错误、用时、sandbox id |
| `outputs/e2b_eval/summary.json` | 汇总：ok=0 / failed=1（不在单个 case 目录内） |
| `.../before_agent.log` | 5 条 before 命令、MCP 注册、AGENTS.md 追加 |
| `.../tools_health.json` | 9201/9202 健康检查 |
| `.../before_agent.json` | BlackboxServer 健康、register 响应（key 已脱敏） |
| `.../openclaw_logs.txt` | OpenClaw gateway 日志；19-20 行是 redacted placeholder，21-26 行是 401 |
| `.../service_logs.txt` | 两个工具服务日志 |
| `e2b_eval/.env` | 当前运行配置（含密钥，勿提交/外传） |
| `e2b_eval/.env.example` | 模板；第 11 行是占位符 `sk-...` |

### 6.2 脱敏日志摘录

```text
# error.txt（最后一行）
e2b_eval.blackbox.BlackboxError: agent turn ... unknown:
Backend request failed: openclaw /v1/chat/completions returned 401:
{"error":{"message":"401 Authentication Fails, Your api key: ****-... is invalid", ...}}
```

```text
# openclaw_logs.txt
02:21:37 ... Config last-known-good promotion skipped: redacted secret placeholder at models.providers.deepseek.apiKey
02:21:56 ... Config last-known-good promotion skipped: redacted secret placeholder at models.providers.deepseek.apiKey
02:21:59 ... embedded run agent end: ... model=deepseek-flash provider=deepseek
             error=LLM request failed. rawError=401 Authentication Fails, Your api key: ****-... is invalid
02:21:59 ... model fallback decision: ... reason=auth next=none
```

```text
# before_agent.log（节选）
## start_services (rc=0, timed_out=False, 2347 ms)
{"status":"ok","service":"ivf_similar", ...}
{"status":"ok","service":"ivf_predict", ...}
## register_mcp (rc=0, timed_out=False, 12629 ms)
Saved MCP server "claw-tools" to .../home/.openclaw/openclaw.json.
## expose_tool_instructions (rc=0, timed_out=False, 601 ms)
append_agents_md: appended 2 tool(s) to .../AGENTS.md
```

### 6.3 时间线

| 时间 | 事件 |
|---|---|
| 2026-10-27 09:30:49 | 真实 OpenClaw 0974 开始运行（shell export 真实 key） |
| 2026-10-27 09:31:47 | 该次运行成功：score 0.975，3 次 MCP 调用，agent 10.0s |
| 2026-10-27 09:36 | 新增 `run.sh`（自动加载 `.env`，不覆盖已有环境变量） |
| 2026-10-28 02:08:07 | `.env` 创建/编辑：只改了 E2B 地址，agent key 仍是 `sk-...` |
| 2026-10-28 02:21:06 | 本次 0974 开始 |
| 2026-10-28 02:21:56 前 | MCP 注册成功，工具说明写入 AGENTS.md |
| 2026-10-28 02:21:56 | OpenClaw 热重载 MCP 配置 |
| 2026-10-28 02:21:59 | DeepSeek 返回 401 |
| 2026-10-28 02:22:00 | 病例标记 failed，写出 `summary.json` |

### 6.4 开发期早期失败对照（均已在上一轮解决）

| 现象 | 原因 | 处理 |
|---|---|---|
| `The sandbox was not found`（commands.run / get_host） | 自托管 E2B 未设置 `E2B_SANDBOX_URL`，SDK 无法路由命令到沙箱 | 在 `.env` 设置 client proxy（如 `http://<host>:3002`）；`config.py` 提前报错 |
| smoke 脚本 `ImportError: cannot import name 'Filesystem'` | 早期临时脚本使用了不存在的 SDK 导出 | 改用正式 `Sandbox`/`commands.run` 路径；该脚本未进入项目 |
| OpenClaw `Invalid config ... agents.defaults` | 模板/adapter 的兼容模式启动重试 | 等 `gateway ready`；README 已记录为正常重试 |
| 每条命令 stderr 的 `source: not found` | `/bin/sh -lc` 执行含 bash 专用 `source` 的 profile | 不影响 rc，README 已记录 |

### 6.5 相关命令

```bash
# 只校验样本和文件映射，不创建沙箱
bash e2b_eval/run.sh --samples runs/openclaw_dev/samples.jsonl --dry-run --limit 2

# 无 LLM 的流水线自检（不消耗模型额度）
bash e2b_eval/run.sh --samples runs/openclaw_dev/samples.jsonl --agent-mode mock --case 0974

# 真实 OpenClaw 单例
bash e2b_eval/run.sh --samples runs/openclaw_dev/samples.jsonl --case 0974 --no-resume

# 单元测试
python3 -m unittest discover -s e2b_eval/tests -v
```
