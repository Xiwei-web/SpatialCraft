# MemP GPT-5.4-mini ViewSpatial 运行记录

- 提交时间（UTC）：2026-09-14T10:30:54.010778+00:00
- Slurm 作业：240349；首次检查状态：PENDING（Priority）。
- 模型：gpt-5.4-mini；reasoning effort：medium；每次输出上限：16384（含推理 token）。
- 数据集：ViewSpatial；environment 2856 题，deployment 2856 题。
- 数据划分：/l/users/xiwei.liu/spatialcraftLog/preparation/viewspatial_seed42_v1；{'algorithm': 'sorted categories; sorted task IDs; one Python random.Random(42); shuffle each category; odd remainders alternate environment first', 'category_key': 'question_type', 'environment_fraction': 0.5, 'seed': 42, 'unit': 'task_instance_not_unique_image'}。
- 流程：environment → build → deployment；每题采集 1 条环境轨迹，仅成功轨迹建库，部署冻结。
- Top-k：3；记忆表示：trajectory + script（proceduralization）；embedding：text-embedding-3-large；每题最多 50 个工具步骤。
- 恢复/强制收尾同为 medium / 16384；所有模型请求省略 temperature、top_p、seed。
- Slurm 资源：1 × A100 40GB、8 CPU、64GB 主存、最长 48 小时。

运行目录：/home/xiwei.liu/spatialcraftRuns/memp_gpt54mini_medium16384_viewspatial_20260914_v1

代码快照：/home/xiwei.liu/spatialcraftSnapshots/memp_gpt54mini_medium16384_viewspatial_20260914_v1

启动脚本：/home/xiwei.liu/spatialcraftRuns/memp_gpt54mini_medium16384_viewspatial_20260914_v1/job.sbatch

日志：/home/xiwei.liu/spatialcraftRuns/memp_gpt54mini_medium16384_viewspatial_20260914_v1/slurm-240349.out

提交前全量 ViewSpatial 数据、公开/私有评分配对及图像哈希预检通过，工具资源无缺失。代码与已通过 125 项离线测试的四集作业快照逐文件一致，使用独立只读副本。本作业独立于四数据集作业 240247，使用不同的输出目录与记忆库。

查询：squeue -j 240349。进度：运行目录下 viewspatial/progress/；最终部署指标：viewspatial/results/deployment.json。首次记录时作业尚在排队，尚无该实验的模型推理结果。
