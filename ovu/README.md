# ovu：获卵数预测

用促排治疗数据预测获卵数的分布。代码在 `task/services/ovu/`（计算）和 `task/services/ovu_services/`（HTTP 服务），这样沙箱里的工具服务能直接找到；本目录只是一个入口，让 `python -m ovu.…` 在仓库根目录下可用。配置文件在 `task/services/ovu/config/`。

## 代码结构

```
task/services/ovu/
  config/           data.yaml（字段、诊断类别、划分）  model.yaml（特征、分组、正则）
                    similar.yaml（相似病例）  db.yaml（病例库）
  pipeline/         clean.py 清洗  split.py 划分（7:1:2，测试集封存）  stats.py 汇总统计
  ml/               features.py 特征编码  model.py softmax 回归与指标  train.py 训练、选 L2、5 折交叉验证
                    evaluation.py 评估与可解释分析  evaluate_test.py 测试集一次性评估  predict.py 预测接口
  similar/          相似病例检索
  db/               病例库：database.py（SQLite，当前默认）  mysql.py（MySQL）  build.py 建库
  report/           population.py 人群统计 Word  model.py 模型性能与可解释分析 Word
task/services/ovu_services/   HTTP 工具服务（预测、相似病例、SQL）和 MCP 工具规格
ovu/mcp.example.json          MCP 配置示例（经 task/runtime/claw_tools_mcp.py 走 MCP 协议）
tests/test_ovu_*.py           57 个测试，全部用模拟数据
```

流程：原始 csv → `pipeline`（清洗、统计、划分）→ `ml` / `similar` / `db` 各自从训练集构建 → `ovu_services` 把三者包成 HTTP 工具，再经 MCP 交给 agent → `report` 只读汇总 JSON 出 Word 文档。所有字段、类别和参数都在 `config/` 里，代码里不写死。

## 数据清洗、统计与划分

```bash
pip install -r task/services/ovu/requirements.txt          # pandas、pyyaml
python -m ovu.pipeline /eds-data/home/wangzhiyuan/01.data/ovulation_induction_12k.csv --out outputs/ovu
```

输出（`outputs/` 已在 .gitignore 中，逐患者文件不提交）：

| 文件 | 内容 |
|---|---|
| `train.csv`、`val.csv` | 清洗后的周期，字段名已统一，不含周期号 |
| `sealed/test.csv` | 固定的测试集，只在最终评估时用一次，不进任何工具 |
| `sealed/test_ids.txt` | 测试集周期号的哈希；有这个文件时，重跑不会改变测试集 |
| `report.md`、`report.json` | 只含汇总统计，可以分享 |

## 配置（`task/services/ovu/config/data.yaml`）

脚本不写死任何字段，全部由配置决定：

- **`columns`**：每个字段一项，写明中文名（`name`）、原始列名（`source`）、类型、合法范围、0 是否表示缺失、类别归并，以及在人群统计文档里归哪张表（`group`：`clinical` 基础情况，`medication` 用药）。
  - 新增字段只需加一项，它会自动进入统计报告、Word 文档（按 `group` 放进对应的表）、数据库和相似病例的可选字段。
  - 删掉一项，这个字段就不再加载，后面所有步骤都跟着去掉，不需要改代码。
  - 数据库（`db.yaml`）的列类型由这里的 `type` 决定：int → INTEGER，float → REAL，category → TEXT；列说明用 `name`。
- **`split_columns`**：把一列文字拆成数值（血压 → 收缩压、舒张压）。
- **`diagnosis`**：诊断是自由文本，先按关键词归成 22 个细类（0/1，一个周期可属于多类，一类都没命中的记为“其他”），再按 `groups` 合并成 10 个病因大类（命中任一细类即为 1）。模型、相似病例和病例库默认用大类加原发/继发不孕；细类保留在统计报告里核对关键词。关键词不区分大小写。报告里的“诊断高频词”和“未被任何关键词覆盖的诊断词”用来修订关键词表；后者也会列出已命中别的类别、但仍有词没被覆盖的周期数。
- **`label`**：获卵数的合法范围和分组边界。
- **`split`**：种子和 test、val 的比例；按获卵数分组分层。

## 清洗规则

- 不看结局来决定任何特征值。超出范围、非数字、应记缺失的 0 都改为缺失，不删行。
- 只有三种情况删行：周期号重复、获卵数缺失、获卵数超范围。
- 周期号只用于去重和划分，不出现在任何输出文件里。

## 划分规则

每个周期的归属只由 `sha256(seed:周期号)` 决定，与行顺序无关。数据里没有患者编号，所以是按周期划分的。

## 获卵数分组模型（`ovu/ml`）

```bash
python -m ovu.ml.train --data outputs/ovu            # 读 train.csv 和 val.csv，不读测试集
python -m ovu.ml.evaluate_test --data outputs/ovu    # 最终评估：测试集只用这一次
python -m ovu.report.model --model-dir outputs/ovu/model     # 生成 模型性能与可解释分析.docx
```

训练做三件事：

1. **选正则强度**：`model.l2` 里的每个值都在训练集上训练、在验证集上比 RPS，取最好的。
2. **5 折交叉验证**：在训练集内，每次用 4 折训练、在剩下 1 折上评估，给出各指标的均值 ± 标准差。这 5 个分折模型同时保存下来，训练 agent 时用（病例看不到自己所在折训练出的模型）。
3. **可解释分析**：置换重要性（打乱某个特征后 RPS 上升多少）、各输入对预期获卵数的影响（系数效应）、部分依赖（只改一个特征时平均预期获卵数怎么变）。

| 输出 | 内容 |
|---|---|
| `model/model.json` | 模型文件：系数和编码用的统计量，不含病例 |
| `model/model_fold<k>.json` | 不含第 k 折训练的模型 |
| `model/model_report.md`、`.json` | 正则强度选择、训练集/验证集/交叉验证指标（对比基线）、校准、置换重要性、系数效应、部分依赖 |
| `model/test_report.json` | 测试集指标、混淆矩阵、校准、测试集上的置换重要性（`evaluate_test` 写） |
| `model/模型性能与可解释分析.docx` | Word 文档：模型与数据、模型性能（测试集为主，附验证集和交叉验证）、可解释分析 |

**测试集只用一次。** `evaluate_test` 每次运行都记在 `sealed/test_evaluations.json`（时间和模型文件的哈希，没有病例信息）；已经有记录时会拒绝再跑，确实要再评估得加 `--again`，报告里会写明是第几次使用。所以先把特征、分组、正则强度都定下来，最后再跑它。还没跑之前，Word 文档会用验证集代替并注明；模型重新训练后，旧的 `test_report.json` 与新模型的哈希对不上，文档同样不用它，并写明当前模型尚未在测试集上评估。

`config/model.yaml` 控制：

- **`label.bins`**：获卵数分组边界。
- **`features`**：列了哪个字段，模型就用哪个，没有阶段之分。Gn 总用量、Gn 总天数、GnRHa 总用量和总天数默认没列，因为它们是促排结束后的累计量；要用的话加一行即可。
  - **`numeric`**：数值特征，`transform` 选 `standard` 或 `log`，可加缺失指示列；缺失用训练集中位数填补。
  - **`categorical`**：类别特征，独热编码；少见、没见过和缺失的取值归入“其他”。
  - **`binary`**：0/1 特征：原发、继发不孕和 10 个病因大类（`dxg_` 开头）；也可以写细类（`dx_` 开头）。训练集中为 1 的周期少于 `binary_min_positive`（默认 20）例的不进模型，报告和 Word 文档里列出被去掉的类别和例数。
- **`model`**：多分类逻辑回归（softmax），只依赖 numpy；`l2` 写一个数或一组候选值。
- **`explain`**：置换重要性重复次数、部分依赖取点数。

模型输出每个分组的概率。评估用适合有序分组的指标：RPS、加权 kappa、相邻组以内的准确率，以及预期获卵数的平均绝对误差。基线是“对所有人都给出训练集的分组占比”。

预测（工具服务会用到）：

```python
from ovu.ml.predict import Predictor
Predictor.from_file("outputs/ovu/model/model.json").predict(age=38, amh=0.9, protocol="拮抗剂方案")
```

返回各分组概率、最可能的分组、预期获卵数，以及每项输入的影响。

## 相似病例检索（`ovu/similar`）

```bash
python -m ovu.similar.build --data outputs/ovu   # 生成 similar/similar.json（只含权重和尺度）
```

`config/similar.yaml` 控制比对哪些特征、各自的权重，以及数值特征的尺度（训练集标准差或固定数值）。距离是各特征差异的加权平均，只算查询里提供了的特征。

```python
import json
from ovu.ml.train import read_rows
from ovu.similar.index import SimilarIndex
index = SimilarIndex(json.load(open("outputs/ovu/similar/similar.json")), read_rows("outputs/ovu/train.csv"))
index.query({"age": 38, "amh": 0.9, "protocol": "拮抗剂方案"}, k=50)
```

返回最相似的 k 例的获卵数分组占比、中位数和四分位数，以及最相似的几位患者（不含编号）。`hide_fold` 用于训练 agent：隐藏当前病例所在的折（`train.csv` 的 `cv_fold` 列），病例看不到自己的结局。

## 病例数据库（`ovu/db`，MySQL）

目前 `db.yaml` 里是 `backend: sqlite`：服务器上还没有 MySQL，工具启动时把 `train.csv` 读进内存建表，查询方式和限制都一样。接上 MySQL 后把它改成 `mysql`，按下面的步骤建库导入即可，不用改代码。

```bash
python -m pip install --user pymysql
export OVU_MYSQL_HOST=127.0.0.1 OVU_MYSQL_PORT=3306
export OVU_MYSQL_ADMIN_USER=<有建库权限的账号> OVU_MYSQL_ADMIN_PASSWORD=<密码>
export OVU_MYSQL_PASSWORD=<给只读账号设的密码>
python -m ovu.db.build --data outputs/ovu        # 建库、建表、导入训练集、建只读账号，并用只读账号把查询示例跑一遍
python -m ovu.db.build --data outputs/ovu --schema-only      # 只生成 db.json 和 schema.sql，不连数据库
```

建库完全由配置决定：列来自 `data.yaml`（删掉一个字段，库里就没有这一列），列类型由 `type` 决定（int → INT，float → DOUBLE，category → VARCHAR，诊断类别 → TINYINT），列注释是 `name`。`config/db.yaml` 控制暴露哪些列、库名和只读账号名、查询限制和查询示例。

| 库 | 内容 | 谁能读 |
|---|---|---|
| `ovu_base` | 底表 `cases_all`（训练集，含折号）和 `fields` | 只有建库账号 |
| `ovu` | 视图 `cases`（全部训练病例，不含折号）和 `fields` | 只读账号 `ovu_reader` |
| `ovu_f0` … `ovu_f4` | 同上，但去掉第 k 折 | 只读账号 `ovu_reader_f0` … `ovu_reader_f4` |

每个只读账号只能 SELECT 自己的视图库，读不到底表、别的折和服务器上的其他库；工具服务这边还限制为单条 SELECT、超时中止（会话级 `max_execution_time`）、最多返回 200 行。训练 agent 时设 `OVU_HIDE_FOLD=k`，工具就换用 `ovu_reader_fk`，病例看不到自己所在的折。测试集不入库，表里没有周期号，默认也没有诊断原文。

`db/schema.sql` 是全部建表语句，末尾注释里有建只读账号的语句。建库账号没有建用户的权限时，可以把这几句交给数据库管理员执行。

| 接口 | 内容 |
|---|---|
| `sql(statement)` | 一条只读 SELECT；超时中止；最多返回 200 行 |
| `overview()` | 数据概览：病例数、获卵数分组占比、每一列的含义、类型、缺失率和分布。不放进 prompt，由 agent 按需调用 |
| `examples()` | 常用查询示例，可以直接执行 |

不连 MySQL 时（`OVU_DB_BACKEND=sqlite`），工具在内存里用 `train.csv` 建同样的表，测试用的就是这种方式。

## 工具服务（`task/services/ovu_services`）

五个工具在一个进程、一个端口（默认 9211），路径是 `POST /tools/<工具名>`，沿用原有服务的日志、调用审计和并发上限。

| 工具 | 作用 |
|---|---|
| `ovu_predict_oocytes` | 各分组概率、最可能的分组、预期获卵数、每项输入的影响 |
| `ovu_similar_cases` | 最相似的 k 例的获卵数分布 |
| `ovu_case_sql` | 病例库只读 SQL |
| `ovu_data_overview` | 病例库的数据概览 |
| `ovu_sql_examples` | 常用查询示例 |

```bash
OVU_DATA_DIR=outputs/ovu PYTHONPATH=task/services python task/services/ovu_services/server.py      # 启动
curl -s -X POST localhost:9211/tools/ovu_predict_oocytes -H 'content-type: application/json' -d '{"age": 38, "amh": 0.9}'
OVU_DATA_DIR=outputs/ovu PYTHONPATH=task/services python -m ovu_services.specs --tools task/ovu_tools   # 生成 MCP 工具规格
```

前两个工具接收哪些参数、参数的范围和说明，都来自模型文件和检索配置（也就是 yaml）；改了配置并重新构建后，重启服务、重新生成规格即可，不用改服务代码。`OVU_HIDE_FOLD=<折号>` 用于训练 agent：预测用没见过该折的模型，检索和病例库去掉该折的病例。

### 通过 MCP 调用

agent 不直接调 HTTP，而是走 MCP 协议，和妊娠工具一样用仓库里现成的 MCP 服务器 `task/runtime/claw_tools_mcp.py`（stdio，不依赖第三方包）：

```
agent（OpenClaw 等 MCP 客户端） ──MCP stdio──▶ claw_tools_mcp.py ──HTTP──▶ ovu_services（:9211）
                                      读 task/ovu_tools/*.json      每次调用记一行 dispatches.jsonl
```

它从 `OMNI_TOOL_SPECS_DIR` 读工具规格，`tools/list` 列出 5 个工具，`tools/call` 按规格里的 `backend.url` 转发到 HTTP 服务；参数不合法、SQL 被拒时返回 `isError: true` 和原因。妊娠工具是 `claw-tools`（读 `task/tools/`），OVU 单独注册成 `ovu-tools`（读 `task/ovu_tools/`），两者互不影响。

```bash
cd /home/wangzhiyuan/ivf_clinical_tools
OVU_DATA_DIR=outputs/ovu PYTHONPATH=task/services python -m ovu_services.specs --tools task/ovu_tools          # 1. 生成规格
OVU_DATA_DIR=outputs/ovu PYTHONPATH=task/services python task/services/ovu_services/server.py &                # 2. 启动 HTTP 服务
OMNI_MCP_SERVER_NAME=ovu-tools OMNI_TOOL_SPECS_DIR=task/ovu_tools python task/runtime/claw_tools_mcp.py --list-tools   # 3. 检查：列出工具
OMNI_MCP_SERVER_NAME=ovu-tools OMNI_TOOL_SPECS_DIR=task/ovu_tools OMNI_TRACE_DIR=outputs/ovu_tool_trace \
  python task/runtime/claw_tools_mcp.py --call-tool ovu_predict_oocytes --arguments '{"age": 38, "amh": 0.9}'  #    检查：调一次
```

在 MCP 客户端里注册：Claude Desktop、MCP Inspector 等按 `ovu/mcp.example.json` 配置（把 `<仓库绝对路径>` 换掉）；OpenClaw 用

```bash
openclaw mcp set ovu-tools '{"command": "python3", "args": ["/home/wangzhiyuan/ivf_clinical_tools/task/runtime/claw_tools_mcp.py"], "env": {"OMNI_MCP_SERVER_NAME": "ovu-tools", "OMNI_TOOL_SPECS_DIR": "/home/wangzhiyuan/ivf_clinical_tools/task/ovu_tools", "OMNI_TRACE_DIR": "/home/wangzhiyuan/ivf_clinical_tools/outputs/ovu_tool_trace", "NO_PROXY": "127.0.0.1,localhost"}}'
```

agent 看到的工具名带服务器前缀，如 `ovu-tools__ovu_predict_oocytes`。HTTP 服务不在 9211 时设 `OMNI_TOOL_PORT_MAP=9211:<端口>`。调用记录 `dispatches.jsonl` 含病例参数，放在 `outputs/` 下，不提交。`tests/test_ovu_mcp.py` 用同样的 stdio 消息测 initialize、tools/list、tools/call 和出错返回。

## 人群统计 Word 文档（`ovu/report`）

```bash
pip install python-docx matplotlib               # 只有生成 Word 文档需要
python -m ovu.report.population --report outputs/ovu/report.json     # 生成 outputs/ovu/人群统计分析.docx
```

只读 `report.json`（汇总统计），不读病例，所以可以在任何机器上生成。内容：研究人群与清洗、年龄分布、获卵数分布、基础情况与内分泌、用药方案与用药、不孕因素与诊断、数据处理说明，共 9 张表、5 张图。换数据后重跑流程再执行这一条，所有数字和图表都会更新。

## 完整流程

```bash
python -m ovu.pipeline /eds-data/home/wangzhiyuan/01.data/ovulation_induction_12k.csv --out outputs/ovu   # 清洗、统计、划分
python -m ovu.report.population --report outputs/ovu/report.json   # 人群统计 Word 文档
python -m ovu.ml.train --data outputs/ovu                # 模型、交叉验证、可解释分析
python -m ovu.similar.build --data outputs/ovu           # 检索配置
python -m ovu.db.build --data outputs/ovu                # MySQL 病例库
python -m ovu.ml.evaluate_test --data outputs/ovu        # 最终评估（只跑一次）
python -m ovu.report.model --model-dir outputs/ovu/model # 模型性能与可解释分析 Word 文档
```

测试：`python -m unittest tests.test_ovu_pipeline tests.test_ovu_ml tests.test_ovu_similar tests.test_ovu_db tests.test_ovu_services tests.test_ovu_mcp tests.test_ovu_report`（用模拟数据，不需要真实数据）。MySQL 部分另有测试，设置 `OVU_TEST_MYSQL_ADMIN_USER`、`OVU_TEST_MYSQL_ADMIN_PASSWORD` 后才运行，会在服务器上建 `ovu_test*` 库和账号。
