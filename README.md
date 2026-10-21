> template origin: https://huggingface.co/datasets/huang3eng/Dressage-Claw
> template task: stock-analysis-five-dimension_1eecda49d411（已替换为 IVF 任务 `ivf-similar-patients-and-prediction-age-amh`）

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
| 目标一：相似患者检索 `ivf_similar_patients` | 已完成：独立服务（端口 9201），URL 测试通过 |
| 目标一：妊娠预测 ML `ivf_predict_outcome` | 已完成：35 项指标训练、预测只用三项输入，训练与预测分离，独立服务（端口 9202），URL 测试通过 |
| 目标二：Claude Code 自主调用、分析、回溯 | 已完成：Claude Code 自主调用两个服务并比较结果，调用轨迹写入 `dispatches.jsonl` |
| 服务实现 | FastAPI + Pydantic；计算在固定大小的线程池中异步执行，并发有上限；日志写入文件并滚动；检索策略和返回字段可扩展 |

### 仓库结构（本仓库即一个任务目录）

```markdown
├── task.json                        # 任务元数据、两个服务、grader 配置、文件映射
├── task/
│   ├── prompt.md                    # 示例任务：虚构患者 38 岁、AMH 0.9、Gn 10 天，要求使用两个工具
│   ├── scripts/                     # start_services.sh 分别启动两个服务；stop_services.sh；setup.sh
│   ├── tools/                       # ivf_similar_patients.json（→9201）、ivf_predict_outcome.json（→9202），由代码生成
│   ├── services/
│   │   ├── ivf_services/            # HTTP 服务（FastAPI），结构同模板 mock_services/<name>/server.py
│   │   │   ├── similar/server.py    #   服务一：相似患者（9201）：请求模型、接口、读训练数据
│   │   │   ├── predict/server.py    #   服务二：妊娠预测（9202）：请求模型、接口、只读模型文件
│   │   │   ├── _base.py             #   两个服务共用：日志、有上限的计算线程池、audit/reset/health、错误处理
│   │   │   └── specs.py             #   由请求模型生成 task/tools 工具规格和 system prompt 的工具部分
│   │   ├── clinical_tools/          # 计算逻辑，只依赖 numpy，不涉及 Web
│   │   │   ├── common.py            #   公共定义
│   │   │   ├── cleaning.py          #   结局盲清洗（去泄露、去异常值）
│   │   │   ├── similar.py           #   相似患者检索：可替换的检索策略 + 统一的门槛与统计
│   │   │   ├── predict.py           #   逻辑回归：train_model（35 项训练）与 Predictor（三项输入预测）
│   │   │   └── train_predictor.py   #   训练脚本：训练数据 → 模型文件
│   │   └── models/lr_35_indicators.json  # 训练好的模型文件（只含系数、协方差、均值等汇总量）
│   └── runtime/                     # claw_tools_mcp.py、gen_trace.py、patch_llm_judge.py（模板原样）、system_prompt_prefix.txt
├── grader/                          # 模板打分框架原样；grader.py 中只替换了 GRADING_SPEC
├── tests/test_clinical_tools.py     # 单元测试（合成数据）
├── mcp.example.json                 # Claude Code 的 MCP 配置示例
└── requirements.txt
```

已删除模板中与本任务无关的文件：股票工单示例的 mock 服务（config / contacts / helpdesk）及其 fixtures、内置 yaml 库（含 macOS 专用 `.so`）、5 个示例工具规格、未被引用的 `grader/task.yaml`。

### 数据

- 甘肃省妇幼保健院 IVF-ET 公开数据（Mendeley x5s3ky9cpv），患者级 1415 例；**只使用训练集 1132 例**，283 例封存测试集不进入任何工具和模型训练。
- 病例数据不在仓库中。相似患者服务通过 `IVF_TRAIN_CSV` 读取；任务容器内默认路径 `/tmp/omni_services/data/train_clean.csv`。
- 结局盲清洗（规则不读标签）：`python -m clinical_tools.cleaning --input <训练集CSV> --out-dir <新目录>`
  - 去泄露：18 列按结局回填的类别均值（如 AMH 3.11/3.39、空腹血糖 5.05/5.04、男方PR 28.63/29.11），无论结局一律置为缺失；
  - 异常值：精液常规全为 0、男方血肌酐 <20（疑似尿素）、PRL>200、血细胞压积 >60、百分比超出 0–100，置为缺失；
  - 输出 `train_clean.csv` 和只含计数的 `cleaning_summary.json`；行数不是 1132 时拒绝运行。

### 服务一：相似患者检索 `ivf_similar_patients`（端口 9201）

- 输入：`age`（女方年龄，岁，18–55）、`amh`（ng/mL，0–50）；可选 `age_window`（1–3，默认 1）、`strategy`（检索策略，默认 `age_amh_band`）、`extra_fields`（相似患者列表中额外返回的指标，中文指标名，最多 10 个，默认不返回）。参数名必须是英文：API 只接受字母、数字、`_ . -`，中文参数名会被 Claude Code 跳过。
- 相似人群（策略 `age_amh_band`）：年龄 ±`age_window` 岁，且 AMH 同一档（<1.2 / 1.2–3.5 / >3.5 ng/mL，1.2 为 POSEIDON 界值）。
- **门槛：符合条件少于 15 例时拒绝返回统计**，只返回例数和放宽建议，不返回妊娠率和个体结局。
- 达到门槛时返回：例数、临床妊娠/未妊娠数、妊娠率与 95% Wilson 置信区间、训练集总体妊娠率、年龄和 AMH 的中位数（范围）、相似患者列表（年龄、AMH、`extra_fields` 指定的指标、结局，按接近程度排列，最多 50 例；清洗中置空的值为 null）；AMH=0.02 的病例单独标注（可能为检测下限）。
- 不返回病例编号，也不用编号排序（本数据集编号顺序本身对应结局）。
- 为 agentic RL 预留 `IVF_EXCLUDE_CASE`：评估某训练病例时，检索看不到该病例所在的 1/5 数据。

### 服务二：妊娠预测 `ivf_predict_outcome`（端口 9202）

**训练与预测分离：**

```bash
# 训练（离线，一次）：训练数据 → 模型文件
PYTHONPATH=task/services python -m clinical_tools.train_predictor --csv <训练集CSV> --out task/services/models/lr_35_indicators.json
# 预测服务：只读模型文件，不需要训练数据
IVF_MODEL=task/services/models/lr_35_indicators.json python task/services/ivf_services/predict/server.py
```

- **训练**（与项目之前的 LR 相同）：全部 35 项指标，缺失用训练集均值填补，标准化，L2 正则逻辑回归（C=1，不做类别加权）；用 numpy 牛顿法拟合，与 sklearn `LogisticRegression(C=1)` 的系数一致（最大差 5×10⁻⁷）。
- **预测**：只接收 `age`（女方年龄）、`amh`（AMH）、`gn_days`（Gn 用药总天数）；其余 32 项指标，以及这三项中没有提供的，都用训练集均值代替。三项至少提供一个。
- **输出只有两项**：`{"妊娠概率": 0.267, "95%置信区间": [0.217, 0.324]}`。置信区间是由系数协方差得到的 Wald 区间（logit 尺度），只反映模型估计的不确定性，不是个体结局的范围。
- 模型文件 `models/lr_35_indicators.json`：35 项指标的均值与标准差、截距与系数、协方差矩阵、训练样本数、交叉验证指标、训练数据 SHA256；不含任何逐病例记录。已有同名文件时训练脚本拒绝覆盖。
- 模型表现（1132 例，5 次重复 5 折交叉验证，均值、标准差、系数都在每折训练部分重新计算）：

  | 使用方式 | AUC | Brier |
  |---|---|---|
  | 预测时给全部 35 项 | 0.590 | 0.2384 |
  | **预测时只给年龄、AMH、Gn 天数（本工具）** | **0.593** | **0.2326** |
  | 对照：一律预测总体妊娠率 | 0.5 | 0.2384 |

  只给三项时并不比给全部指标差：其余指标的作用很弱，且多为噪声。
- 三项输入在原始尺度上的系数（对数几率）：年龄 −0.061/岁，AMH +0.015/(ng/mL)，Gn +0.211/天。
- 注意：Gn 用药总天数在数据中只有 8–12 天五个取值，疑似被截断；年龄线性进入，对 37 岁以后妊娠率加速下降刻画不足。
- agentic RL 时如需排除被评估病例，用 `train_predictor --exclude-case <ID>` 另训一个模型文件。

### 服务实现

两个服务都是 FastAPI 应用，每个 `server.py` 的结构与模板 `mock_services/helpdesk/server.py` 相同：Pydantic 请求模型（参数、类型、范围、默认值一目了然）→ 接口函数 → 调用记录 → `uvicorn.run`。共用部分在 `_base.py`。

- **参数校验（Pydantic）**：类型、范围、未知参数（`extra="forbid"`）、可选值都在请求模型里声明；不合法时返回 422 和可读的说明，例如 `参数校验失败：age: Input should be less than or equal to 55`，Agent 可以据此改正后重试。`task/tools/*.json` 由 `python -m ivf_services.specs` 从请求模型生成，参数只在一处定义。
- **异步与并发上限**：接口是 `async` 的，检索、预测这类计算交给固定大小的线程池执行，不阻塞事件循环。同时计算的数量上限为 `IVF_MAX_CONCURRENCY`（默认 4），排队上限为 `IVF_MAX_PENDING`（默认 512，排队不占线程）；再多的请求立即返回 429（带 `Retry-After`）。连接数上限为 `IVF_MAX_CONNECTIONS`（默认 1024），超出返回 503。线程数不随请求数增长，不会出现请求一多就开出大量线程或进程的情况。
- **日志**：设置 `IVF_LOG_DIR` 后，每个服务写 `<服务名>.log`，按大小滚动（`IVF_LOG_MAX_BYTES` 默认 10 MB，保留 `IVF_LOG_BACKUPS` 默认 5 份）；`start_services.sh` 默认写到 `$OMNI_TASK_DIR/logs/`。内容包括：启动时加载的数据或模型、每个请求一行（方法、路径、状态码、耗时）、每次工具调用一行（参数和结果摘要，不记录病例明细）、繁忙拒绝的警告、异常的完整报错（同时返回 500）、停止时按状态码统计的请求数。`IVF_LOG_LEVEL` 控制级别（DEBUG 时也记录 /health 探测）。
- **状态查询**：`GET /health` 返回运行时长、各状态码的请求数、正在计算和排队的数量、加载的数据或模型；`GET /ivf/audit` 返回调用记录（内存中最多 `IVF_AUDIT_MAX` 条，默认 2000），`POST /ivf/reset` 清空。
- **错误注入**：与模板相同，`ERROR_RATE>0` 时随机返回 429/500 或延迟，用于测试 Agent 的重试能力；默认关闭。

**扩展方式**（不需要改服务）：

- 新的检索策略：在 `clinical_tools/similar.py` 写一个 `SimilarityStrategy` 子类（`name`、`description`、`select()` 返回匹配的病例和接近程度），加入 `STRATEGIES`；请求参数 `strategy` 的可选值、工具说明会随之更新（重新运行 `ivf_services.specs`）。门槛、统计、排序、返回格式对所有策略通用。策略需要新参数时，在 `Query` 和请求模型中各加一个字段。
- 返回更多患者信息：调用时在 `extra_fields` 里列出指标即可（35 项清洗后的指标都可选）；需要新增一类信息（如分组统计）时，在策略的 `profile()` 里加。

**压力测试**（1132 例真实数据，相似检索服务，同时发出的请求数）：

| 同时请求数 | 旧实现（标准库 ThreadingHTTPServer） | 新实现（FastAPI，默认上限） |
|---|---|---|
| 300 | 253 个成功，47 个连接被直接重置（无任何返回） | 300 个全部成功，0.9 秒；服务进程线程数始终为 9 |
| 2000 | — | 979 个成功，超出连接上限的 1021 个立即返回 503；线程数仍为 9，之后服务正常 |

### 运行

需要 Python ≥ 3.10。

```bash
pip install -r requirements.txt   # numpy、fastapi、pydantic、uvicorn
# 1. 清洗训练集（只需一次）
PYTHONPATH=task/services python -m clinical_tools.cleaning --input xgb_train_1132.csv --out-dir outputs/clean_v1
# 2. 分别启动两个服务（模型文件默认用 task/services/models/lr_35_indicators.json）
IVF_TRAIN_CSV=outputs/clean_v1/train_clean.csv bash task/scripts/start_services.sh
# 3. URL 测试
curl -X POST http://127.0.0.1:9201/ivf/similar_patients -H 'Content-Type: application/json' -d '{"age": 38, "amh": 0.9}'
curl -X POST http://127.0.0.1:9202/ivf/predict_outcome -H 'Content-Type: application/json' -d '{"age": 38, "amh": 0.9, "gn_days": 10}'
# 每个服务另有：GET /health，GET /ivf/audit（调用记录），POST /ivf/reset；接口文档：http://127.0.0.1:9201/docs
# 日志：/tmp/omni_task/logs/ivf_similar.log、ivf_predict.log（OMNI_TASK_DIR 或 IVF_LOG_DIR 可改）
# 4. 停止
bash task/scripts/stop_services.sh
```

Windows（在仓库根目录开两个 PowerShell 窗口，各运行一个服务；环境变量只对当前窗口有效）：

```powershell
# 窗口 1：相似患者检索（9201）
$env:IVF_TRAIN_CSV = "outputs\clean_v1\train_clean.csv"; $env:IVF_LOG_DIR = "outputs\logs"
python task\services\ivf_services\similar\server.py
# 窗口 2：妊娠预测（9202），默认读取 task\services\models\lr_35_indicators.json
$env:IVF_LOG_DIR = "outputs\logs"
python task\services\ivf_services\predict\server.py
```

### 接入 Claude Code

1. 把 `mcp.example.json` 中的 `<仓库绝对路径>` 换成实际路径（Windows 用正斜杠），另存为 `.mcp.json`；`command` 建议写 Python 的完整路径。
2. 后端为 DeepSeek 等非 Anthropic 模型时，先设置 `ENABLE_TOOL_SEARCH=false`，让工具定义直接发给模型。
3. 运行：`claude -p "$(cat task/prompt.md)" --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__ivf-clinical__ivf_similar_patients" "mcp__ivf-clinical__ivf_predict_outcome"`
4. 回溯：每次调用写入 `OMNI_TRACE_DIR/dispatches.jsonl`（工具名、请求的服务地址、参数、返回、耗时、时间）。

### 打分（grader）

`grader/grader.py` 的 `GRADING_SPEC` 针对 `task/prompt.md`：必须调用两个工具；确定性部分检查关键数值（17.4%、37.1%、26.7%、39.2%）是否出现在回答中；LLM judge 部分评估统计与预测是否如实转述、两种证据的一致性与总体比较、门槛与窗口判断、局限说明、不给治疗建议。期望数值基于 1132 例训练集经上述清洗后的结果和 `models/lr_35_indicators.json`。

### 测试

```bash
pip install httpx   # 只有测试需要（FastAPI 的 TestClient）；fastapi 低于 0.112 时用 "httpx<0.28"
PYTHONPATH=task/services:tests python -m unittest discover -s tests
```

已在 Python 3.10 + requirements.txt 中的最低版本、Python 3.13 + 最新版本两种环境下通过。

覆盖：
- 清洗：不读标签且可重复、两组回填值与异常值规则。
- 相似检索：拒绝门槛、统计与手工计数一致、输出不含编号且与行顺序无关、排除折、`extra_fields`、新增一个检索策略类即可被服务使用。
- 预测：逻辑回归能恢复已知系数、AUC 与成对定义一致、模型文件只含汇总量、输出只有概率和置信区间、其余指标与缺失输入用训练均值、训练脚本不覆盖已有模型、删除训练数据后仍能预测。
- 服务：Pydantic 校验（缺参数、类型、范围、未知参数、未知策略和指标、非法 JSON）返回 422；audit/reset/health；每个服务只提供自己的端点；请求和调用写入日志、日志文件按大小滚动；40 个并发请求压向一个慢速、上限很小的服务时，超出部分返回 429，工作线程不超过上限，服务保持可用。
- 工具规格与 system prompt 与请求模型一致、端口正确；MCP 桥接调用两个服务，参数错误时返回 isError。

## Results

数据：1132 例训练集清洗后，有年龄和 AMH 的可检索病例 1045 例，总体临床妊娠率 39.2%。

相似患者检索：

| 查询 | 相似人群 | 例数 | 结果 |
|---|---|---|---|
| 38 岁，AMH 0.9 | 37–39 岁，AMH <1.2 | 23 | 妊娠 4 例，17.4%（95% CI 7.0%–37.1%） |
| 44 岁，AMH 0.5 | 43–45 岁，AMH <1.2 | 8 | 拒绝统计（<15） |
| 44 岁，AMH 0.5，窗口 ±2 | 42–46 岁，AMH <1.2 | 15 | 妊娠 0 例，0%（95% CI 0–20.4%） |
| 44 岁，AMH 0.5，窗口 ±3 | 41–47 岁，AMH <1.2 | 16 | 妊娠 0 例，0%（95% CI 0–19.4%） |

妊娠预测（`models/lr_35_indicators.json`）：

| 输入 | 妊娠概率 | 95% CI |
|---|---|---|
| 38 岁，AMH 0.9，Gn 10 天 | 0.267 | 0.217–0.324 |
| 38 岁，AMH 0.9（Gn 用均值） | 0.271 | 0.221–0.328 |
| 44 岁，AMH 0.5，Gn 9 天 | 0.169 | 0.115–0.241 |
| 30 岁，AMH 2.5，Gn 12 天 | 0.482 | 0.426–0.538 |
| 只提供 38 岁 | 0.279 | 0.229–0.336 |

Claude Code 端到端（2026-10-20）：
- 只有相似检索时：Windows 本地 Claude Code + DeepSeek 后端与 Linux 上 Claude 后端的调用序列相同。38 岁病例调用 1 次；44 岁病例首次被拒绝（8 例），随后自主把窗口放宽到 ±2、±3 岁重查。两台机器（数据分别由原始训练集和 v2 训练集清洗得到）返回的工具结果完全一致。
- 两个独立服务（Claude 后端，`task/prompt.md`）：分别调用 9201 的检索（23 例，17.4%）和 9202 的预测（26.7%，95% CI 21.7%–32.4%），回答指出模型区间落在相似人群区间内、点估计高约 9 个百分点、两者都低于总体 39.2%，并说明了模型预测时只用三项输入、区分能力有限等局限；所有数字与工具返回一致。
- 此前版本（预测输出含模型卡片）在 DeepSeek 后端下的回答也逐条核对过，引用的数字全部来自工具返回。

Grader 自检：上述两服务任务的真实回答在确定性部分得满分（LLM judge 部分需配置 judge 运行）；只调用一个工具会扣掉必需工具覆盖分；不调用工具、编造概率并给出用药建议的回答得 0 分。

### 后续

- 更多任务（如例数不足、需要放宽窗口的病例），用于 agentic RL。
- 胚胎冷冻数等条件：字段定义确认前不加入（可能泄露结局）。
