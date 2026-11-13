# ovu：获卵数预测

用促排治疗数据预测获卵数的分布。代码在 `task/services/ovu/`（计算）和 `task/services/ovu_services/`（HTTP 服务），这样沙箱里的工具服务能直接找到；本目录只是一个入口，让 `python -m ovu.…` 在仓库根目录下可用。配置文件在 `task/services/ovu/config/`。

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

- **`columns`**：每个字段一项，写明中文名（`name`）、原始列名（`source`）、类型、合法范围、0 是否表示缺失、类别归并、预测时点（`baseline` 进周期前 / `plan` 启动方案 / `during` 促排过程中）。
  - 新增字段只需加一项，它会自动进入统计报告、Word 文档（按预测时点放进对应的表）、数据库和相似病例的可选字段。
  - 删掉一项，这个字段就不再加载，后面所有步骤都跟着去掉，不需要改代码。
  - 数据库（`db.yaml`）的列类型由这里的 `type` 决定：int → INTEGER，float → REAL，category → TEXT；列说明用 `name`。
- **`split_columns`**：把一列文字拆成数值（血压 → 收缩压、舒张压）。
- **`diagnosis`**：诊断是自由文本，按关键词归成若干类别（0/1）。报告里的“诊断高频词”和“未匹配比例”用来修订关键词表。
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
python -m ovu.ml.train --data outputs/ovu        # 读 train.csv 和 val.csv，不读测试集
cat outputs/ovu/model/model_report.md
```

| 输出 | 内容 |
|---|---|
| `model/model.json` | 模型文件：系数和编码用的统计量，不含病例 |
| `model/model_report.md`、`.json` | 训练集和验证集指标（对比基线）、置换重要性、各列对预期获卵数的影响 |

`config/model.yaml` 控制：

- **`label.bins`**：获卵数分组边界。
- **`stages`**：允许哪些预测时点的字段进入模型。默认 `baseline` 和 `plan`（启动前预测）。
- **`features.numeric`**：数值特征，`transform` 选 `standard` 或 `log`，可加缺失指示列；缺失用训练集中位数填补。
- **`features.categorical`**：类别特征，独热编码；少见、没见过和缺失的取值归入“其他”。
- **`features.binary`**：0/1 特征（诊断类别）。
- **`model`**：目前是多分类逻辑回归（softmax），只依赖 numpy。

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

## 病例数据库（`ovu/db`）

```bash
python -m ovu.db.build --data outputs/ovu        # 生成 db/db.json 和 db/overview.json，并把每条查询示例实际执行一遍
```

agent 用只读 SQL 自己探索训练病例。`config/db.yaml` 控制暴露哪些列、列的中文说明、查询限制和查询示例。

| 接口 | 内容 |
|---|---|
| `sql(statement)` | 一条只读 SELECT；拒绝写入、ATTACH、PRAGMA 和多条语句；超时中止；最多返回 200 行 |
| `overview()` | 数据概览：病例数、获卵数分组占比、每一列的含义、类型、时点、缺失率和分布。不放进 prompt，由 agent 按需调用 |
| `examples()` | 常用查询示例，可以直接执行 |

表里没有周期号和折号，默认也没有诊断原文（自由文本）。存储是每个沙箱里的内存 SQLite：这样训练 agent 时可以用 `hide_fold` 隐藏当前病例所在的折，集中部署的数据库做不到这一点。测试集不入库。

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

## 人群统计 Word 文档（`ovu/report`）

```bash
pip install python-docx matplotlib               # 只有这一步需要
python -m ovu.report.population --report outputs/ovu/report.json     # 生成 outputs/ovu/人群统计分析.docx
```

只读 `report.json`（汇总统计），不读病例，所以可以在任何机器上生成。内容：研究人群与清洗、年龄分布、获卵数分布、基础情况与内分泌、用药方案与用药、不孕因素与诊断、数据处理说明，共 9 张表、5 张图。换数据后重跑流程再执行这一条，所有数字和图表都会更新。

## 完整流程

```bash
python -m ovu.pipeline <原始.csv> --out outputs/ovu      # 清洗、统计、划分
python -m ovu.ml.train --data outputs/ovu                # 模型和 5 个分折模型
python -m ovu.similar.build --data outputs/ovu           # 检索配置
python -m ovu.db.build --data outputs/ovu                # 病例库配置和概览
```

测试：`python -m unittest tests.test_ovu_pipeline tests.test_ovu_ml tests.test_ovu_similar tests.test_ovu_db tests.test_ovu_services tests.test_ovu_report`（用模拟数据，不需要真实数据）。
