# GPT-5.4-mini tool-only ReAct baseline

2026-09-10 已提交任务 **227082**。使用项目既有测试输入，四个数据集共925题，每题一条trajectory、单pass。实时状态以Slurm和progress.json为准。

| 数据集 | 测试题数 |
|---|---:|
| RoboSpatial | 175 |
| ERQA | 200 |
| Omni3D | 250 |
| SAT | 300 |

实际输入逐字节复用 `gpt54mini_direct_baseline_20260910_v1/data/`，三个空间数据集为既有deployment划分，SAT为既有300题。没有environment训练阶段，没有Experience/Skill检索、种子Skill、embedding、知识更新或PPO。

## 模型与交互

配置 `configs/models/gpt-5.4-mini-react.yaml`：gpt-5.4-mini、reasoning=none、temperature=0、每次正常调用最多4096输出tokens。API不设置生成seed。允许简短文字说明与一个工具action；模型证据充分时可直接最终作答，不强制每题一定调用工具。

每条trajectory是多轮LLM决策→一个工具执行→observation→下一次LLM调用。最多50次工具交互；完整截断action继续执行，不完整时当前step最多一次1024-token action recovery。Recovery不消耗额外工具step。50次交互耗尽后只允许一次512-token最终作答，不再执行工具。解析/恢复失败的trajectory记零分；基础设施错误停止该数据集、保存断点并继续其他独立数据集，不记作错误答案。

提供与正式方案相同的11个工具：`detect`、`segment`、`mask`、`geometry`、`scale`、`reconstruct`、`pose`、`graph`、`motion`、`ocr`、`draw`。重型工具分别使用现有GroundingDINO、SAM3、MoGe2、DepthAnything3、OrientAnything、EasyOCR本地权重。单张A100 40GB、8CPU、64GB主机内存、48小时；工具逐次加载并释放权重，四个数据集依次执行。

模型接收原图全部视图、原始顺序、detail=high；每轮反馈工具文字/结构化结果、生成图像，以及JSON artifact内容。工具输入只允许当前题公开图像及当前trajectory已产生的artifact，私有评分标签不会进入模型/工具上下文。沿用纯模型baseline的选项标签和答案格式、现有确定性verifier及分类accuracy。

## 代码与验证

入口：`python -m spatialcraft.experiments.run_react_baseline`，默认仅离线检查，`--execute`启用正式评测。启动脚本 `scripts/inference/gpt54mini_tool_react.sbatch`。实际任务执行运行目录中的冻结 `code_snapshot/`，不受后续工作区修改影响。

新增API ReAct接线和独立runner；通用journal支持random_seed=None，ContextComposer可显式关闭知识提示。修复Responses适配器的助手历史文本编码为output_text；原先第二轮会因input_text而收到400。ReAct专用解析保留原始API响应，将max_output_tokens截断映射给既有action recovery。工具schema保留原有可选参数，API侧设strict=false，继续本地参数验证。接口依据：[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)。

验证记录：

- 完整回归202项通过（40.30秒）；最终API历史修复后针对性回归63项通过（2.60秒），Ruff及shell语法通过。
- 真实GPT-5.4-mini合成图验收通过：geometry→draw→图像observation→最终答案；三次正常模型调用，返回gpt-5.4-mini-2026-03-17，reasoning_tokens=0。此验收强制工具顺序仅用于测试接口，正式评测使用auto。
- GPU验收任务227055：六个重型工具均成功，工具输出序列化及artifact落盘通过，使用environment图像，不计入测试题成绩。
- 冻结离线检查通过：同一925题、私有标签隔离、原图hash、工具源码与35份工具资源文件hash、库版本和生成配置。

验证详情保存在 `/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_tool_react_baseline_20260910_v1/validation/`；完整验收过程位于 `/l/users/xiwei.liu/spatialcraftLog/validation/gpt54mini_tool_react_20260910_v1/`。早期失败的合成API检查保留审计，最终通过记录为api_roundtrip_v2。

## 结果与逐步记录

运行目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_tool_react_baseline_20260910_v1`。

- `{dataset}/progress.json`：该数据集实时完成数、accuracy、分类结果及工具/模型调用统计。
- `{dataset}/results/deployment.json`：完整数据集结束后的总体与分类accuracy。
- `{dataset}/results/predictions.jsonl`：逐题答案、reward、失败原因、模型用量和工具调用统计。
- `results/summary.json`：已完成数据集结果与失败列表，所有数据集结束后为最终汇总。
- `{dataset}/stages/tasks/NNNNN/steps/`：每步模型请求/原始响应、action、工具结果、observation和恢复调用。
- `{dataset}/stages/tasks/NNNNN/trajectory/result.json`：完整trajectory。
- `{dataset}/tool_store/`：工具图像、深度、分割、几何等artifact。
- `launcher_logs/slurm-227082.log`：正式任务日志。
- `submission.json`、`preflight.json`：提交配置和资源指纹。

```bash
squeue -j 227082
tail -f /l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_tool_react_baseline_20260910_v1/launcher_logs/slurm-227082.log
```

重启同一冻结命令和输出目录会验证并复用已经提交的模型/工具步骤。API已返回但尚未提交journal时崩溃，可能重复调用，不保证exactly-once计费。
