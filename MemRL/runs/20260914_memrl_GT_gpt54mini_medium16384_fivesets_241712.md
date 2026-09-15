# MemRL-GT 五集单作业提交记录

作业 ID：241712
提交时间（UTC）：2026-09-14T18:38:57.565397+00:00
最近核查时间（UTC）：2026-09-14T18:40:14.654773+00:00
当前状态：PENDING；原因或节点：(QOSMaxJobsPerUserLimit)

严格按用户要求，仅提交一个 Slurm 作业。该作业在同一进程依次执行 RoboSpatial、ERQA、Omni3D、SAT、ViewSpatial。

## 参数

- 变体：MemRL-GT。
- Actor / reflection：gpt-5.4-mini；reasoning effort=medium；max_output_tokens=16384。
- 恢复和强制最终回答沿用16384；temperature/top_p/seed不发送。
- 环境单轮、每题一次采样、max_steps=50；环境后冻结最终记忆再部署。
- candidate_k=10、top_k=3、utility_weight=0.5、learning_rate=0.3、Q_init=0。
- embedding=text-embedding-3-large，阈值由公开环境query两两cosine的80%分位点校准。
- GT答案仅提供给环境反思，actor和部署不接收当前任务GT，部署不学习。

## 数据规模

| 数据集 | 环境 | 部署 |
| --- | ---: | ---: |
| robospatial | 175 | 175 |
| erqa | 200 | 200 |
| omni3d | 251 | 250 |
| sat | 300 | 300 |
| viewspatial | 2856 | 2856 |

## 资源与验证

1 × A100-SXM4-40GB、8 CPU、64 GB、48 小时。

五集数据、图像、公开/私有评分配对及工具路径预检通过。本轮81项反思/runner/CLI专项测试通过；相同源码此前完整MemRL 124项测试通过。

GT快照与R快照代码哈希完全一致，227个源码文件在提交前逐一校验。

源码哈希：fee1158543f0634cac617b6ecca21367714f650292741357040148847fd9a6d6

快照：/home/xiwei.liu/spatialcraftSnapshots/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1

## 产物

- 输出目录：/home/xiwei.liu/spatialcraftRuns/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1
- 作业脚本：/home/xiwei.liu/spatialcraftRuns/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1/job.sbatch
- Slurm日志：/home/xiwei.liu/spatialcraftRuns/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1/slurm-241712.out
- 总体状态：/home/xiwei.liu/spatialcraftRuns/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1/results/all_summary.json
- 各集部署结果：/home/xiwei.liu/spatialcraftRuns/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1/GT/<dataset>/results/deployment.json
- 各集冻结记忆：/home/xiwei.liu/spatialcraftRuns/memrl_GT_gpt54mini_medium16384_fivesets_20260914_v1/GT/<dataset>/memory/frozen.json
