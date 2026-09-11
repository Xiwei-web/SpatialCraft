# ViewSpatial RAG：GPT-5.4-mini medium / 16384

2026-09-11 16:32（Dubai）提交任务 **229312**，与四数据集任务229287独立并行。提交时PENDING；实时状态以 `squeue -j 229312` 为准。

## 设置

- 环境与部署均为gpt-5.4-mini、reasoning_effort=medium、单次最多16384 tokens（包含reasoning）。
- 环境集2856题，单pass、每题4次生成，共11424次；测试集2856题，每题1次生成。
- 沿用seed42分类内50/50划分，奇数余项environment优先交替分配；不是在全部5712题上评测。
- text-embedding-3-large，cosine Top-3不同历史任务；只提供原始任务文本与历史模型输出，不写反思/lessons，不调用空间工具。
- 不传temperature、top_p和generation seed；服务端采样设置，不能表述为temperature=0。
- 4 CPUs、16GB内存、72小时，无GPU；不依赖229287完成。

## 核验

完整执行源码与229287的冻结快照逐文件一致，复用此前34项离线回归和真实API验收证据；本次没有重复执行模型API烟雾测试。ViewSpatial重新完成全部输入和媒体的离线检查，公开输入与私有评分输入均与此前mini直接回答baseline逐字节相同，跨集合task ID无重叠，未发现完全相同的图片/问题/选项跨集合重复。保留每题全部原图，最多32张。

## 路径

根目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_rag_viewspatial_medium16384_20260911_v1`。

- `submission.json`：任务号与资源配置。
- `run_config.json`、`preflight.json`：实际模型、预算和数据范围。
- `slurm-229312.out`、`slurm-229312.err`：标准输出与错误日志。
- `viewspatial/progress.json`：当前阶段进度。
- `viewspatial/environment/records/`：持续新增的环境原始记录。
- `viewspatial/memory/records.jsonl`、`memory/index.json`、`memory/snapshot.json`：最终冻结记录库与索引。
- `viewspatial/stages/`：逐调用模型输入/输出、检索和评分记录。
- `viewspatial/results/deployment.json`：完成后的accuracy。
- `viewspatial/results/predictions.jsonl`：逐题回答、评分及检索来源。
- `results/accuracy.md`、`results/summary.json`：完成后的汇总。
- `validation/checks.json`：输入一致性与复用验证证据。

入口为项目 `RAG/gpt54mini_viewspatial_medium16384.sbatch`，已单独复制到运行目录 `launcher_code/`；执行源码位于 `code_snapshot/`。部署阶段不更新记录库，结果与229287完全分开。
