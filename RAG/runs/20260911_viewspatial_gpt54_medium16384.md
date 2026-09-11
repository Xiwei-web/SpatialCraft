# ViewSpatial RAG：GPT-5.4 medium / 16384

2026-09-11 17:09（Dubai）提交任务 **229405**。提交时状态 **PENDING / QOSMaxJobsPerUserLimit**：CPU队列的每用户并发运行名额已满，已有229287、229312、229347运行。名额释放后由Slurm自动调度，无需重复提交；实时状态以 `squeue -j 229405` 为准。

## 设置

- GPT-5.4，环境建库和部署均 reasoning_effort=medium，单次最多16384 tokens（包含reasoning）。
- ViewSpatial固定environment/deployment各2856题，沿用seed42分类内50/50划分。
- 环境单pass、每题4次，共11424次生成；部署每题1次，共2856次生成。
- text-embedding-3-large，cosine Top-3不同历史任务，原始task/output示例检索。
- 不发送temperature、top_p、generation seed；不启用空间工具，不写反思、lessons、Experience或Skill。
- GPT-5.4从空记录库独立建库，不复用mini模型输出或记忆。
- 4 CPUs、16GB内存、72小时，无GPU。作业时间限制从开始运行后计算。

## 验证

执行代码与229312的冻结快照逐文件一致，全部环境、公开测试与私有评分输入逐字节一致，公开测试输入也因此与此前ViewSpatial直接回答baseline相同。模型和模型配置hash是相对229312唯一变化的运行配置项。输入/媒体预检查通过，跨集合task ID不重叠，未发现完整图片/问题/选项内容重复。每题保留全部原图。

复用同一代码的34项离线回归与229347已通过的GPT-5.4 medium/16384真实建库→embedding检索→多图部署验收证据；本次没有重复调用合成验收API。

## 路径

根目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54_rag_viewspatial_medium16384_20260911_v1`。

- `submission.json`：提交信息；`run_config.json`、`preflight.json`：实际设置与数据范围。
- 启动后 `slurm-229405.out`、`slurm-229405.err`：标准输出和错误日志。
- 启动后 `viewspatial/progress.json`：阶段进度。
- `viewspatial/environment/records/`：环境阶段逐条原始记录。
- `viewspatial/memory/records.jsonl`、`memory/index.json`、`memory/snapshot.json`：最终冻结记忆与索引。
- `viewspatial/stages/`：逐调用输入、输出、检索和评分日志。
- 完成后 `viewspatial/results/deployment.json`：数据集accuracy。
- 完成后 `viewspatial/results/predictions.jsonl`：逐题回答与评分。
- 完成后 `results/accuracy.md`、`results/summary.json`：汇总。
- `validation/checks.json`：输入一致性和复用验证记录。

项目入口 `RAG/gpt54_viewspatial_medium16384.sbatch`，提交副本在运行目录 `launcher_code/`；实际代码位于 `code_snapshot/`，不会受后续工作区源码修改影响。
