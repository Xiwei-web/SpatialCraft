# MemRL-R 提交记录（2026-09-14）

模型：gpt-5.4-mini；reasoning effort：medium；单次 max_output_tokens：16384。actor、反思与恢复调用使用相同预算，temperature/top_p/seed 不发送。

本次只执行 R，环境反思不提供正确答案。每个环境任务采样一次、环境单轮；环境学习后冻结最终记忆再进行部署。

| 作业 | 数据集 | 最新状态 | 原因或节点 |
| --- | --- | --- | --- |
| 241512 | robospatial, erqa, omni3d, sat | PENDING | (Priority) |
| 241513 | viewspatial | PENDING | (Priority) |

每个作业：1 × A100-SXM4-40GB、8 CPU、64 GB、48 小时。两个作业独立运行。

## 数据规模

| 数据集 | 环境 | 部署 |
| --- | ---: | ---: |
| robospatial | 175 | 175 |
| erqa | 200 | 200 |
| omni3d | 251 | 250 |
| sat | 300 | 300 |
| viewspatial | 2856 | 2856 |

## 配置与校验

检索 candidate_k=10、top_k=3、utility_weight=0.5；Q 学习率 0.3、新记忆 Q=0。相似度阈值由公开环境 query 的两两 cosine 80% 分位点校准。每次轨迹最多 50 步，embedding=text-embedding-3-large。

124 项 MemRL 离线测试通过；五集输入/图像/私有评分配对与工具路径预检通过。提交前逐文件核对快照哈希。

代码快照：/home/xiwei.liu/spatialcraftSnapshots/memrl_R_gpt54mini_medium16384_20260914_v1

源码哈希：fee1158543f0634cac617b6ecca21367714f650292741357040148847fd9a6d6

## 产物

- 作业 241512 输出：/home/xiwei.liu/spatialcraftRuns/memrl_R_gpt54mini_medium16384_foursets_20260914_v1
- 日志：/home/xiwei.liu/spatialcraftRuns/memrl_R_gpt54mini_medium16384_foursets_20260914_v1/slurm-241512.out
- 汇总：/home/xiwei.liu/spatialcraftRuns/memrl_R_gpt54mini_medium16384_foursets_20260914_v1/results/all_summary.json
- 作业 241513 输出：/home/xiwei.liu/spatialcraftRuns/memrl_R_gpt54mini_medium16384_viewspatial_20260914_v1
- 日志：/home/xiwei.liu/spatialcraftRuns/memrl_R_gpt54mini_medium16384_viewspatial_20260914_v1/slurm-241513.out
- 汇总：/home/xiwei.liu/spatialcraftRuns/memrl_R_gpt54mini_medium16384_viewspatial_20260914_v1/results/all_summary.json

最近状态核查时间（UTC）：2026-09-14T17:09:55.372859+00:00
