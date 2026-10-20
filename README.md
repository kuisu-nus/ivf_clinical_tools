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
| 目标一：妊娠预测 ML `ivf_predict_outcome` | 已完成：训练与预测分离，独立服务（端口 9202），URL 测试通过 |
| 目标二：Claude Code 自主调用、分析、回溯 | 已完成：Claude Code 自主调用两个服务并比较结果，调用轨迹写入 `dispatches.jsonl` |

### 仓库结构（本仓库即一个任务目录）

```markdown
├── task.json                        # 任务元数据、两个服务、grader 配置、文件映射
├── task/
│   ├── prompt.md                    # 示例任务：虚构患者 38 岁、AMH 0.9、Gn 10 天，要求使用两个工具
│   ├── scripts/                     # start_services.sh 分别启动两个服务；stop_services.sh；setup.sh
│   ├── tools/                       # ivf_similar_patients.json（→9201）、ivf_predict_outcome.json（→9202），由代码生成
│   ├── services/
│   │   ├── clinical_tools/          # 只依赖 numpy
│   │   │   ├── common.py            #   公共定义与参数校验
│   │   │   ├── cleaning.py          #   结局盲清洗（去泄露、去异常值）
│   │   │   ├── similar.py           #   相似患者检索逻辑
│   │   │   ├── similar_service.py   #   服务一：相似患者（9201），读训练数据
│   │   │   ├── predict.py           #   逻辑回归：train_model（训练）与 Predictor（预测）
│   │   │   ├── train_predictor.py   #   训练脚本：训练数据 → 模型文件
│   │   │   ├── predict_service.py   #   服务二：妊娠预测（9202），只读模型文件
│   │   │   ├── http_service.py      #   两个服务共用的 HTTP 框架
│   │   │   └── specs.py             #   生成 task/tools 下的工具规格
│   │   └── models/lr_age_amh_gn.json  # 训练好的模型文件（只含系数、协方差、均值等汇总量）
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

- 输入：`age`（女方年龄，岁）、`amh`（ng/mL）、可选 `age_window`（1–3，默认 1）。参数名必须是英文：API 只接受字母、数字、`_ . -`，中文参数名会被 Claude Code 跳过。
- 相似人群：年龄 ±`age_window` 岁，且 AMH 同一档（<1.2 / 1.2–3.5 / >3.5 ng/mL，1.2 为 POSEIDON 界值）。
- **门槛：符合条件少于 15 例时拒绝返回统计**，只返回例数和放宽建议，不返回妊娠率和个体结局。
- 达到门槛时返回：例数、临床妊娠/未妊娠数、妊娠率与 95% Wilson 置信区间、训练集总体妊娠率、年龄和 AMH 的中位数（范围）、相似患者列表（年龄、AMH、结局，按接近程度排列，最多 50 例）；AMH=0.02 的病例单独标注（可能为检测下限）。
- 不返回病例编号，也不用编号排序（本数据集编号顺序本身对应结局）。
- 为 agentic RL 预留 `IVF_EXCLUDE_CASE`：评估某训练病例时，检索看不到该病例所在的 1/5 数据。

### 服务二：妊娠预测 `ivf_predict_outcome`（端口 9202）

**训练与预测分离：**

```bash
# 训练（离线，一次）：训练数据 → 模型文件
PYTHONPATH=task/services python -m clinical_tools.train_predictor --csv <训练集CSV> --out task/services/models/lr_age_amh_gn.json
# 预测服务：只读模型文件，不需要训练数据
PYTHONPATH=task/services python -m clinical_tools.predict_service --model task/services/models/lr_age_amh_gn.json
```

- 变量：只用 `age`（女方年龄）、`amh`（AMH）、`gn_days`（女方 Gn 用药总天数），线性进入逻辑回归；用 numpy 牛顿法拟合（与 sklearn 结果一致）。
- 缺失处理：训练和预测时，缺失的变量都用训练集均值代替（年龄 30.50，AMH 3.53，Gn 10.09；训练集中 AMH 缺失 87 例）。预测时三个变量至少提供一个。
- **输出只有两项**：`{"妊娠概率": 0.278, "95%置信区间": [0.23, 0.333]}`。置信区间是系数协方差得到的 Wald 区间（logit 尺度），只反映模型估计的不确定性，不是个体结局的范围。
- 模型文件 `models/lr_age_amh_gn.json`：系数、协方差矩阵、填补用均值、训练样本数、交叉验证指标、训练数据 SHA256；不含任何逐病例记录。已有同名文件时训练脚本拒绝覆盖。
- 模型表现（1132 例，5 次重复 5 折交叉验证，每折内重新计算填补均值）：

  | 变量 | AUC | Brier |
  |---|---|---|
  | 年龄 | 0.566 | 0.2350 |
  | 年龄 + AMH | 0.570 | 0.2351 |
  | 年龄 + Gn 天数 | 0.595 | 0.2318 |
  | **年龄 + AMH + Gn 天数（采用）** | **0.594** | **0.2323** |
  | 对照：一律预测总体妊娠率 | 0.5 | 0.2384 |

  系数（对数几率）：截距 −0.817，年龄 −0.056/岁，AMH +0.023/(ng/mL)，Gn +0.197/天。
- 注意：Gn 用药总天数在数据中只有 8–12 天五个取值，疑似被截断；年龄线性进入，对 37 岁以后妊娠率加速下降刻画不足（曾试过加入年龄²与 log AMH，AUC 0.601，按要求改为三变量线性模型）。
- agentic RL 时如需排除被评估病例，用 `train_predictor --exclude-case <ID>` 另训一个模型文件。

### 运行

```bash
pip install -r requirements.txt
# 1. 清洗训练集（只需一次）
PYTHONPATH=task/services python -m clinical_tools.cleaning --input xgb_train_1132.csv --out-dir outputs/clean_v1
# 2. 分别启动两个服务（模型文件默认用 task/services/models/lr_age_amh_gn.json）
IVF_TRAIN_CSV=outputs/clean_v1/train_clean.csv bash task/scripts/start_services.sh
# 3. URL 测试
curl -X POST http://127.0.0.1:9201/ivf/similar_patients -H 'Content-Type: application/json' -d '{"age": 38, "amh": 0.9}'
curl -X POST http://127.0.0.1:9202/ivf/predict_outcome -H 'Content-Type: application/json' -d '{"age": 38, "amh": 0.9, "gn_days": 10}'
# 每个服务另有：GET /health，GET /ivf/audit（调用记录），POST /ivf/reset
# 4. 停止
bash task/scripts/stop_services.sh
```

Windows（两个窗口分别运行）：

```powershell
$env:PYTHONPATH = "task/services"
python -m clinical_tools.similar_service --csv outputs\clean_v1\train_clean.csv
python -m clinical_tools.predict_service --model task\services\models\lr_age_amh_gn.json
```

### 接入 Claude Code

1. 把 `mcp.example.json` 中的 `<仓库绝对路径>` 换成实际路径（Windows 用正斜杠），另存为 `.mcp.json`；`command` 建议写 Python 的完整路径。
2. 后端为 DeepSeek 等非 Anthropic 模型时，先设置 `ENABLE_TOOL_SEARCH=false`，让工具定义直接发给模型。
3. 运行：`claude -p "$(cat task/prompt.md)" --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__ivf-clinical__ivf_similar_patients" "mcp__ivf-clinical__ivf_predict_outcome"`
4. 回溯：每次调用写入 `OMNI_TRACE_DIR/dispatches.jsonl`（工具名、请求的服务地址、参数、返回、耗时、时间）。

### 打分（grader）

`grader/grader.py` 的 `GRADING_SPEC` 针对 `task/prompt.md`：必须调用两个工具；确定性部分检查关键数值（17.4%、37.1%、27.8%、39.2%）是否出现在回答中；LLM judge 部分评估统计与预测是否如实转述、两种证据的一致性与总体比较、门槛与窗口判断、局限说明、不给治疗建议。期望数值基于 1132 例训练集经上述清洗后的结果和 `models/lr_age_amh_gn.json`。

### 测试

```bash
PYTHONPATH=task/services:tests python -m unittest discover -s tests
```

覆盖：清洗不读标签且可重复、两组回填值与异常值规则、拒绝门槛、统计与手工计数一致、输出不含编号且与行顺序无关、排除折；逻辑回归能恢复已知系数、AUC 与成对定义一致、模型文件只含汇总量、预测输出只有概率和置信区间、缺失用训练均值、训练脚本不覆盖已有模型、删除训练数据后仍能预测；两个服务各自只提供自己的端点、工具规格与代码一致且端口正确、MCP 桥接调用两个服务。

## Results

数据：1132 例训练集清洗后，有年龄和 AMH 的可检索病例 1045 例，总体临床妊娠率 39.2%。

相似患者检索：

| 查询 | 相似人群 | 例数 | 结果 |
|---|---|---|---|
| 38 岁，AMH 0.9 | 37–39 岁，AMH <1.2 | 23 | 妊娠 4 例，17.4%（95% CI 7.0%–37.1%） |
| 44 岁，AMH 0.5 | 43–45 岁，AMH <1.2 | 8 | 拒绝统计（<15） |
| 44 岁，AMH 0.5，窗口 ±2 | 42–46 岁，AMH <1.2 | 15 | 妊娠 0 例，0%（95% CI 0–20.4%） |
| 44 岁，AMH 0.5，窗口 ±3 | 41–47 岁，AMH <1.2 | 16 | 妊娠 0 例，0%（95% CI 0–19.4%） |

妊娠预测（`models/lr_age_amh_gn.json`）：

| 输入 | 妊娠概率 | 95% CI |
|---|---|---|
| 38 岁，AMH 0.9，Gn 10 天 | 0.278 | 0.230–0.333 |
| 38 岁，AMH 0.9（Gn 用均值） | 0.282 | 0.233–0.337 |
| 44 岁，AMH 0.5，Gn 9 天 | 0.183 | 0.129–0.255 |
| 30 岁，AMH 2.5，Gn 12 天 | 0.481 | 0.428–0.535 |
| 只提供 38 岁 | 0.294 | 0.245–0.350 |

Claude Code 端到端（2026-10-20）：
- 只有相似检索时：Windows 本地 Claude Code + DeepSeek 后端与 Linux 上 Claude 后端的调用序列相同。38 岁病例调用 1 次；44 岁病例首次被拒绝（8 例），随后自主把窗口放宽到 ±2、±3 岁重查。两台机器（数据分别由原始训练集和 v2 训练集清洗得到）返回的工具结果完全一致。
- 两个独立服务（Claude 后端，`task/prompt.md`）：分别调用 9201 的检索（23 例，17.4%）和 9202 的预测（27.8%，95% CI 23.0%–33.3%），回答指出模型概率落在相似人群区间内但点估计高约 10 个百分点、两者都低于总体 39.2%，并说明了模型只用三项变量、区分能力有限等局限；所有数字与工具返回一致。
- 此前版本（预测输出含模型卡片）在 DeepSeek 后端下的回答也逐条核对过，引用的数字全部来自工具返回。

Grader 自检：上述两服务任务的真实回答在确定性部分得满分（LLM judge 部分需配置 judge 运行）；只调用一个工具会扣掉必需工具覆盖分；不调用工具、编造概率并给出用药建议的回答得 0 分。

### 后续

- 更多任务（如例数不足、需要放宽窗口的病例），用于 agentic RL。
- 胚胎冷冻数等条件：字段定义确认前不加入（可能泄露结局）。
