# ViewSpatial：GPT-5.4 deployment 半集 baseline

2026-09-10 提交 Slurm 任务 **227000**，仅评测固定的 **2856 道 deployment 题**，执行一次。实时状态以 Slurm 和 progress.json 为准。

## 配置与输入

- 模型：`gpt-5.4`，配置 `configs/models/gpt-5.4-baseline.yaml`。
- reasoning=none、temperature=0、max_output_tokens=4096，每题一次回答，单 pass，无工具、Experience、Skill。
- 复用共享划分 `/l/users/xiwei.liu/spatialcraftLog/preparation/viewspatial_seed42_v1`；不重新划分，不评测 environment 的2856题。
- 按 question_type 类别内50/50、seed42；奇数余项交替分给 environment/deployment，environment优先。42仅控制划分，API未设置生成seed。
- 输入与 mini 任务226866的 deployment public/private 文件逐字节一致，保留相同题序、全部原图及视图顺序，detail=high。2856为题目记录数，多图题保留全部图片，共8950次图像引用、4101份唯一图像内容、最多32图/题。
- 同一纯视觉提示词和确定性选择题评分。完成后输出整体及各 question_type 的 accuracy。

划分的分类数量和完整说明见 `docs/viewspatial_mini_deployment_baseline_20260910.md`。原有 mini 任务独立保存，本次结果不复用其回答。

## 执行与验证

资源：CPU分区 `cscc-cpu-p`，2CPU、16GB内存、24小时，无GPU；通过 OpenAI Responses API执行。逐题保存journal，可按相同冻结配置恢复。输出截断/拒答按单次回答baseline计零分，不追加recovery；API基础设施错误重试耗尽后停止并保留断点。

使用与mini任务相同的冻结源码和数据，启动时校验源码manifest。启动脚本为 `scripts/inference/viewspatial_gpt54_baseline.sbatch`，实际任务运行目录中的 `launcher_code/viewspatial_gpt54_baseline.sbatch`。

已通过shell语法、冻结源码完整性、离线preflight、2856题唯一ID与标签对应、environment排除、固定划分hash及全部4101份图片存在检查。模型配置确认为gpt-5.4 / reasoning=none / 4096 tokens / temperature=0。此次不修改推理或评分代码。

## 结果与日志

运行目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54_viewspatial_deployment2856_seed42_20260910_v1`。

- `viewspatial/progress.json`：已完成数量及当前accuracy。
- `viewspatial/results/deployment.json`：全部2856题结束后的总体和分类accuracy。
- `viewspatial/results/predictions.jsonl`：逐题答案、reward、模型版本和用量。
- `results/summary.json`：完整运行汇总。
- `launcher_logs/slurm-227000.log`：执行日志。
- `viewspatial/stages/tasks/NNNNN/{model,score}/`：逐题请求、响应及评分记录。
- `submission.json`、`preflight.json`、`validation/input_validation.json`：提交信息、配置和输入检查记录。

```bash
squeue -j 227000
tail -f /l/users/xiwei.liu/spatialcraftLog/runs/gpt54_viewspatial_deployment2856_seed42_20260910_v1/launcher_logs/slurm-227000.log
```
