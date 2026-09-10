# GPT-5.4-mini 纯视觉 baseline

2026-09-10 提交任务 **226209**，已在 CPU 节点 **cn-03** 启动。调度状态以 Slurm 为准。

用户确认：RoboSpatial 175、ERQA 200、Omni3D 250，复用已有 seed42 deployment 划分；SAT 使用当前 `SAT_test_circular_300.parquet` 的全部 300 题。合计 925 题。

## 评测配置

- `gpt-5.4-mini`，Responses API，`reasoning.effort=none`、`temperature=0`、每次最多 4096 output tokens。
- 每题一次独立回答，单 pass；输入仅原始图像、题目和选项。不调用工具，不检索/学习 Experience 或 Skill，不执行 embedding、PPO 或训练。
- 原始图像字节、全部视图、原始顺序，`detail=high`；不经过本地 Qwen 图像降采样。Pointing 要求输出原图归一化坐标。
- 输出为 `Final Answer: ...`。本入口是单回答 baseline，不走多步 agent/action-recovery/forced-final；预算截断、拒答或无答案单独记录并计零分，无额外生成补救。
- 评分复用项目确定性 verifier，报告 binary accuracy 和 question_type/answer_type/source_answer_type 分类；numeric 默认绝对容差 1e-3、相对容差 1e-2，pointing 复用 mask-hit。它是与当前项目一致的评测，不声称所有细节等同于各 benchmark 官方 evaluator。
- API 异常由 SDK 最多重试 3 次；仍失败则停止对应数据集并保留断点，其余数据集继续。基础设施错误不混入错误答案。外部成功但尚未落盘时的崩溃可能导致重试，不保证 exactly-once 计费。
- 四个数据集各一个线程、每个线程顺序请求。CPU 4 核、16GB、24小时，不申请 GPU。

## 位置与查看

运行目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_direct_baseline_20260910_v1`

- `launcher_logs/slurm-226209.log`：持续更新的执行日志。
- `{dataset}/progress.json`：已评测数量、当前准确率、token 用量与状态；未完成时只是部分结果。
- `{dataset}/results/deployment.json`：该数据集完整结果，仅全部题目结束后写入。
- `{dataset}/results/predictions.jsonl`：逐题答案、reward、verifier、实际返回模型版本和 token 用量。
- `results/summary.json`：四数据集结束后的汇总；`failed_datasets` 非空时并未完成全部评测。
- `{dataset}/stages/tasks/NNNNN/{model,score}/`：请求/原始响应/评分及重试记录；模型请求只含公开任务信息，本地媒体引用由固定哈希绑定并在 API 调用时编码成 data URI。
- `data/`：固定 public/private 输入、任务 ID、图像及评分 mask 哈希。GT 只进入本地评分器。
- `code_snapshot/` 与 `launcher_code/`：本次冻结源码及启动入口。`submission.json` 为提交配置，`validation/` 保存验收结果。

```bash
squeue -j 226209
tail -f /l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_direct_baseline_20260910_v1/launcher_logs/slurm-226209.log
```

源码入口：`src/spatialcraft/experiments/run_api_baseline.py`；独立配置 `configs/models/gpt-5.4-mini-baseline.yaml`；Slurm 入口 `scripts/inference/api_baseline.sbatch`。默认命令只准备数据，`--execute` 才调用 API。恢复必须使用冻结代码、同一配置/数据/目录，并确认没有同目录写入作业；已提交模型结果复用，不能修改 journal 绕过绑定。

## 提交前验证

8 项 baseline 关键回归通过（0.63秒），覆盖标签隔离、原图上传、评分、截断和异常处理、API 断点复用及拒绝代码变更恢复；Ruff 和 shell 语法检查通过。925 题请求/媒体格式检查及原始图像哈希校验通过。

真实 API 用 16 张合成红色图像验收，返回 `red`、status=completed、实际模型 `gpt-5.4-mini-2026-03-17`、reasoning_tokens=0。验收不使用 benchmark 标签，也不计入正式题数。已确认正式 CPU 作业开始产生四数据集结果；最终准确率以完整结果文件为准。

参数参考：[OpenAI GPT-5.4-mini 文档](https://developers.openai.com/api/docs/models/gpt-5.4-mini)。API 未提供 seed 参数；temperature=0 不构成服务端逐字确定性的保证，响应中保存实际模型版本。
