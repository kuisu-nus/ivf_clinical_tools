> template origin: https://huggingface.co/datasets/huang3eng/Dressage-Claw
> template task: stock-analysis-five-dimension_1eecda49d411（已替换为 IVF 任务 `ivf-similar-patients-age-amh`）

## Background
基于这个MCP的模板，构建IVF-Clinical-Tools的MCP服务，用于IVF-Agentic根据患者的实际情况进行工具的调用。

目录结构如下：
```markdown
tasks/<task_id>/
├── task.json              # 任务元数据、grader 配置、文件映射
├── task/
│   ├── prompt.md          # 任务指令（prompt 文本）
│   ├── workspace/         # 工作区文件
│   ├── scripts/           # setup.sh、start_services.sh、stop_services.sh
│   ├── tools/             # MCP 工具规格 JSON 文件
│   ├── services/          # Mock 服务配置
│   └── runtime/           # 辅助脚本（MCP、trace、judge 补丁）
└── grader/
    └── run.sh             # Grader 入口（输出 overall_score JSON）
```

目标一: 新增IVF-Clinical-tools：
- 支持相似患者检索:
    - input: 患者的关键字段
    - output: 相关患者的临床信息+妊娠结局分布
- 妊娠结局预测ML
    - input: 当前患者的关键临床字段
    - output: 当前患者的ML预测结局
> 要确保新增的tool是可以通过url进行测试通过, 并部署到本地服务器上

目标二: 通过本地的claude code黑盒，可以实现这些tools的自主调用，分析，回溯。而不是之前的手动将工具调用过程写死。

## Methods

### 当前进度

| 目标 | 状态 |
|---|---|
| 目标一：相似患者检索 `ivf_similar_patients` | 已完成：URL 测试通过，已在本地部署 |
| 目标一：妊娠结局预测 ML | 未开始 |
| 目标二：Claude Code 自主调用、分析、回溯 | 已完成：Claude Code（DeepSeek 后端）自主调用，调用轨迹写入 `dispatches.jsonl` |

### 仓库结构（本仓库即一个任务目录）

```markdown
├── task.json                      # 任务元数据、grader 配置、文件映射（已改为 IVF 任务）
├── task/
│   ├── prompt.md                  # 示例任务：虚构患者 38 岁、AMH 0.9
│   ├── scripts/                   # setup.sh、start_services.sh、stop_services.sh
│   ├── tools/ivf_similar_patients.json   # MCP 工具规格（由代码生成）
│   ├── services/clinical_tools/   # 工具服务：common / cleaning / similar / service（只依赖 numpy）
│   └── runtime/                   # claw_tools_mcp.py、gen_trace.py、patch_llm_judge.py（模板原样）、system_prompt_prefix.txt
├── grader/                        # 模板打分框架原样；grader.py 中只替换了 GRADING_SPEC
├── tests/test_clinical_tools.py   # 单元测试（合成数据）
├── mcp.example.json               # Claude Code 的 MCP 配置示例
└── requirements.txt
```

已删除模板中与本任务无关的文件：股票工单示例的 mock 服务（config / contacts / helpdesk）及其 fixtures、内置 yaml 库（含 macOS 专用 `.so`）、5 个示例工具规格、未被引用的 `grader/task.yaml`。

### 数据

- 甘肃省妇幼保健院 IVF-ET 公开数据（Mendeley x5s3ky9cpv），患者级 1415 例；工具**只使用训练集 1132 例**，283 例封存测试集不进入任何工具。
- 数据文件不在仓库中。部署时通过 `IVF_TRAIN_CSV` 指定；任务容器内默认路径 `/tmp/omni_services/data/train_clean.csv`。
- 结局盲清洗（规则不读标签）：`python -m clinical_tools.cleaning --input <训练集CSV> --out-dir <新目录>`
  - 去泄露：18 列按结局回填的类别均值（如 AMH 3.11/3.39、空腹血糖 5.05/5.04、男方PR 28.63/29.11），无论结局一律置为缺失；
  - 异常值：精液常规全为 0、男方血肌酐 <20（疑似尿素）、PRL>200、血细胞压积 >60、百分比超出 0–100，置为缺失；
  - 输出 `train_clean.csv` 和只含计数的 `cleaning_summary.json`；行数不是 1132 时拒绝运行。

### 工具：`ivf_similar_patients`

- 输入：`age`（女方年龄，岁）、`amh`（ng/mL）、可选 `age_window`（1–3，默认 1）。参数名必须是英文：API 只接受字母、数字、`_ . -`，中文参数名会被 Claude Code 跳过。
- 相似人群：年龄 ±`age_window` 岁，且 AMH 同一档（<1.2 / 1.2–3.5 / >3.5 ng/mL，1.2 为 POSEIDON 界值）。
- **门槛：符合条件少于 15 例时拒绝返回统计**，只返回例数和放宽建议，不返回妊娠率和个体结局。
- 达到门槛时返回：例数、临床妊娠/未妊娠数、妊娠率与 95% Wilson 置信区间、训练集总体妊娠率、年龄和 AMH 的中位数（范围）、相似患者列表（年龄、AMH、结局，按接近程度排列，最多 50 例）；AMH=0.02 的病例单独标注（可能为检测下限）。
- 不返回病例编号，也不用编号排序（本数据集编号顺序本身对应结局）。
- 为 agentic RL 预留 `IVF_EXCLUDE_CASE`：评估某训练病例时，工具看不到该病例所在的 1/5 数据。

### 运行

```bash
pip install -r requirements.txt
# 1. 清洗训练集（只需一次）
PYTHONPATH=task/services python -m clinical_tools.cleaning --input xgb_train_1132.csv --out-dir outputs/clean_v1
# 2. 启动服务
IVF_TRAIN_CSV=outputs/clean_v1/train_clean.csv bash task/scripts/start_services.sh
# 3. URL 测试
curl -X POST http://127.0.0.1:9201/ivf/similar_patients -H 'Content-Type: application/json' -d '{"age": 38, "amh": 0.9}'
# 其他端点：GET /health，GET /ivf/audit（调用记录），POST /ivf/reset
# 4. 停止
bash task/scripts/stop_services.sh
```

Windows：`$env:PYTHONPATH="task/services"; python -m clinical_tools.service --csv outputs\clean_v1\train_clean.csv`，在另一个窗口用 `Invoke-RestMethod` 测试。

### 接入 Claude Code

1. 把 `mcp.example.json` 中的 `<仓库绝对路径>` 换成实际路径（Windows 用正斜杠），另存为 `.mcp.json`；`command` 建议写 Python 的完整路径。
2. 后端为 DeepSeek 等非 Anthropic 模型时，先设置 `ENABLE_TOOL_SEARCH=false`，让工具定义直接发给模型。
3. 运行：`claude -p "$(cat task/prompt.md)" --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__ivf-clinical__ivf_similar_patients"`
4. 回溯：每次调用写入 `OMNI_TRACE_DIR/dispatches.jsonl`（参数、返回、耗时、时间）。

### 打分（grader）

`grader/grader.py` 的 `GRADING_SPEC` 针对 `task/prompt.md`：必须调用 `ivf_similar_patients`；确定性部分检查关键数值（17.4%、37.1%、39.2%）是否出现在回答中；LLM judge 部分评估统计是否如实转述、门槛与窗口判断、局限说明、不给治疗建议。期望数值基于 1132 例训练集经上述清洗后的结果。

### 测试

```bash
PYTHONPATH=task/services:tests python -m unittest discover -s tests
```

覆盖：清洗不读标签且可重复、两组回填值与异常值规则、拒绝门槛、统计与手工计数一致、输出不含编号且与行顺序无关、排除折、HTTP 端点、工具规格与代码一致且参数名合规、MCP 桥接的 tools/list 与 tools/call。

## Results

数据：1132 例训练集清洗后，有年龄和 AMH 的可检索病例 1045 例，总体临床妊娠率 39.2%。

| 查询 | 相似人群 | 例数 | 结果 |
|---|---|---|---|
| 38 岁，AMH 0.9 | 37–39 岁，AMH <1.2 | 23 | 妊娠 4 例，17.4%（95% CI 7.0%–37.1%） |
| 44 岁，AMH 0.5 | 43–45 岁，AMH <1.2 | 8 | 拒绝统计（<15） |
| 44 岁，AMH 0.5，窗口 ±2 | 42–46 岁，AMH <1.2 | 15 | 妊娠 0 例，0%（95% CI 0–20.4%） |
| 44 岁，AMH 0.5，窗口 ±3 | 41–47 岁，AMH <1.2 | 16 | 妊娠 0 例，0%（95% CI 0–19.4%） |

Claude Code 端到端（2026-10-20）：Windows 本地 Claude Code + DeepSeek 后端，以及 Linux 上 Claude Code + Claude 后端，两种后端的调用序列相同：
- 38 岁病例：自主调用 1 次（age=38, amh=0.9）。Claude 后端的回答如实转述了 23 例、17.4% 及置信区间，并判断无需放宽窗口；
- 44 岁病例：首次调用被拒绝（8 例），随后自主把窗口放宽到 ±2、±3 岁重查。Claude 后端的总结说明了样本小、窗口变宽的局限；
- 全部调用均记录在 `dispatches.jsonl`；两台机器（数据分别由原始训练集和 v2 训练集清洗得到）返回的工具结果完全一致。DeepSeek 后端的文字回答尚未逐条核对。

Grader 自检：上述 38 岁病例的真实回答在确定性部分得满分（LLM judge 部分需配置 judge 运行）；不调用工具、编造概率并给出用药建议的回答得 0 分。

### 后续

- 第二个工具：妊娠结局 ML 预测。
- 更多任务（如例数不足、需要放宽窗口的病例），用于 agentic RL。
- 胚胎冷冻数等条件：字段定义确认前不加入（可能泄露结局）。