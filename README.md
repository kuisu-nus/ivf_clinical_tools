> origin: https://huggingface.co/datasets/huang3eng/Dressage-Claw
> origin task: stock-analysis-five-dimension_1eecda49d411

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


## Results