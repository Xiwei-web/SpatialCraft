# ViewSpatial：GPT-5.4-mini deployment 半集 baseline

2026-09-10 提交任务 **226866**。仅执行 ViewSpatial 的 **2856 道 deployment 题**，单轮、每题一次作答。

## 数据划分

此前未发现已保存的 ViewSpatial 半划分。本次对本地 `ViewSpatial-Bench.json` 的5712条题目记录使用项目 `stratified_halves(..., seed=42)`：按question_type类别排序、类别内按task ID排序、使用一个Python Random(42)逐类shuffle、每类50/50；奇数类别余项交替分配，第一次给environment，偶数类别不改变交替顺序。

| question_type | 原始 | environment | deployment（本次评测） |
|---|---:|---:|---:|
| Camera perspective - Object View Orientation | 996 | 498 | 498 |
| Camera perspective - Relative Direction | 1773 | 887 | 886 |
| Person perspective - Object View Orientation | 996 | 498 | 498 |
| Person perspective - Relative Direction | 842 | 421 | 421 |
| Person perspective - Scene Simulation Relative Direction | 1105 | 552 | 553 |
| 合计 | 5712 | 2856 | 2856 |

两半task IDs不相交，所有原始题目恰好分配一次。**5712/2856是题目记录数，不是唯一图片数**：ViewSpatial部分题目需要多张图。测试半集保留每题全部原图与原始顺序，合计8950次图像引用、4101份唯一图像内容、最多32图/题；数据本身在不同题目间复用图像。本划分不是image-disjoint，manifest记录两半共享3090份图像内容。没有向模型发送environment题目进行评测。

共享划分目录：`/l/users/xiwei.liu/spatialcraftLog/preparation/viewspatial_seed42_v1`。

- `manifest.json`：算法、seed42、分类计数、task IDs、原始JSON/ZIP及划分文件hash。
- `viewspatial/splits/deployment.jsonl`：2856题公开输入，已移除GT与私有annotation。
- `viewspatial/verification/deployment.jsonl`：相同题序的本地评分标签。
- `viewspatial/splits/environment.jsonl`、`viewspatial/verification/environment.jsonl`：另2856题留作后续使用，本次不作答。

后续复用会校验源文件和划分hash、协议、半集数量及seed42重算后的成员顺序，不重新随机划分。

## 模型与执行

`gpt-5.4-mini`，reasoning=none、temperature=0、4096-token单次输出上限；每题一次、单pass，无工具/Experience/Skill/embedding/PPO。API未设置生成seed，42仅控制数据划分。输入原图、全部视图、detail=high，选择题输出选项标签。评分沿用项目MultipleChoiceVerifier，输出整体与question_type分类accuracy。

CPU分区cscc-cpu-p，2CPU、16GB、24小时，无GPU。已提交/未完成阶段使用journal断点恢复；输出截断/拒答按单回答baseline计零分，API基础设施错误停止并保留断点，不混入错误答案。不会对截断追加额外模型回答。

## 结果位置

运行目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_viewspatial_deployment2856_seed42_20260910_v1`。

- `viewspatial/progress.json`：实时已完成数量及部分准确率。
- `viewspatial/results/deployment.json`：全部2856题结束后的总体和分类accuracy。
- `viewspatial/results/predictions.jsonl`：每题原始答案、reward、模型版本和用量。
- `results/summary.json`：本次单数据集完整运行汇总。
- `launcher_logs/slurm-226866.log`：执行日志。
- `viewspatial/stages/tasks/NNNNN/{model,score}/`：完整请求、响应和评分审计。
- `data/viewspatial/{public,private}.jsonl`：本次实际评测输入，仅含deployment半集。

```bash
squeue -j 226866
tail -f /l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_viewspatial_deployment2856_seed42_20260910_v1/launcher_logs/slurm-226866.log
```

## 验证与入口

新增共享划分实现 `src/spatialcraft/experiments/viewspatial_split.py`，baseline限定ViewSpatial的evaluation_split=deployment、count=2856、指定split protocol，并检查实际题数与manifest一致；拒绝复用5712题全量缓存。原有四数据集默认选择保持不变。

48项划分、奇数余项、标签隔离、复用、拒绝全量输入、baseline和统计回归通过（0.75秒）。Ruff、shell语法、冻结preflight、全部2856请求格式和原图hash校验通过。单题最大原图内容约3.26MB（base64约4.14MiB）。32张合成图真实API验收通过：返回gpt-5.4-mini-2026-03-17、reasoning_tokens=0。合成验收不计入测试题数。

源码入口仍为 `experiments.run_api_baseline --datasets viewspatial`；正式任务运行冻结的 `code_snapshot/`，启动脚本为 `launcher_code/viewspatial_api_baseline.sbatch`。默认命令只做离线准备，`--execute`才调用API。
