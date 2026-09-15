# GPT-5.4 ViewSpatial RAG：保留断点，剩余环境每题一次

2026-09-12 23:43（Dubai）提交 **233606**，接续原任务 **229405** 的环境记录，提交后 Slurm 确认 **RUNNING / cn-06**。源码冻结校验已通过；截至提交后首次检查，日志尚未输出数据预检查完成或新的模型调用。运行状态不等同于已经越过断点。

## 采样规则

- 原任务已提交375条环境调用结果：前93题各4条，第94题3条。
- 保留全部375条原始输出，不重做已完成调用；第94题已达到新的最低次数，因此不补第4条。
- 其余2762题各生成1次，预计环境库共3137条记录，覆盖全部2856题。
- 测试集仍为固定2856题，每题1次；不评测全部5712题。
- GPT-5.4、reasoning_effort=medium、16384 tokens、text-embedding-3-large、cosine Top-3均沿用原设置。
- 不发送temperature/top_p/generation seed。检索仍按每个历史任务最早完成且非空的输出选择代表，不按正确性筛选。

这是保留既有多次采样记录、剩余题一次采样的续跑，不能标注为“所有环境题均1次”或“所有环境题均4次”。结果报告包含`prior_model_calls`、`new_model_calls`及`rollout_count_histogram`；预检查中的`environment_generation_calls`包含复用记录，`new_environment_generation_calls`才是新增预算。

## 断点导入与验证

新增`RAG/continuation.py`，通过`--resume-environment-from`与`--resume-completed-calls`指定原任务目录和精确连续调用前缀。改变环境采样次数必须使用新输出目录。导入前比对原/新journal，只有源码版本与环境采样次数及续跑来源字段允许不同；逐条检查请求、图片哈希映射、输入绑定和输出校验和。原记录保持原样，校验后在新journal中提交副本。导入过程自身可恢复，已导入记录也不会重新请求模型。

本次代码从229405的原冻结清单重建，原模型provider、配置、提示词、检索、评分和其他依赖逐文件匹配；仅增加/修改RAG续跑相关源码、测试和启动脚本。项目工作区45项相关离线测试通过；实际冻结运行版本使用匹配原依赖版本的测试，共40项通过，其中6项覆盖减少剩余次数、旧记录保留、损坏/缺失记录拒绝、配置差异拒绝及导入再次中断后恢复。未新增模型API烟雾调用。

原输入文件逐字节复用。登录节点在重新读取基准图片时发生存储阻塞，完整媒体重验及375条真实记录导入交由计算节点在生成前执行；不能将离线测试等同于已完成真实断点导入。基准图片和源记录仍位于`/l/users`存储，若该存储持续阻塞，任务也可能停留在启动校验或读取阶段。

## 路径

正式运行目录：`/home/xiwei.liu/spatialcraftRuns/gpt54_rag_viewspatial_remaining1_medium16384_20260912_v1`。

- `submission.json`：233606与229405的继承关系、次数和资源。
- `slurm-233606.out`、`slurm-233606.err`：运行日志。
- `run_config.json`、运行时生成的`preflight.json`：实际配置和预算。
- `viewspatial/stages/environment/import_prefix/result.json`：真实前缀导入完成后的来源和逐调用校验和。
- `viewspatial/progress.json`：当前阶段进度（导入与初始化后开始更新）。
- `viewspatial/environment/records/`：旧记录与新记录组成的环境库。
- `viewspatial/results/environment.json`：环境库次数统计。
- `viewspatial/results/deployment.json`、`results/accuracy.md`：完成后的accuracy。
- `validation/checks.json`、`source_lineage.json`、`frozen_tests.xml`：验证证据。

源码位于共享home的`/home/xiwei.liu/spatialcraftSnapshots/gpt54_rag_viewspatial_remaining1_medium16384_20260912_v1`，正式目录的`code_snapshot`为其链接；这样避开此次Lustre源码复制阻塞。启动脚本为项目`RAG/gpt54_viewspatial_remaining1_medium16384.sbatch`的固定副本，4 CPUs、16GB、72小时，无GPU。

`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54_rag_viewspatial_remaining1_medium16384_20260912_v1`仅为遇到存储阻塞后停止的准备目录，**没有从该目录提交作业，不是233606的结果目录**。原229405目录和原结果保留。

重新提交233606时应保持同一正式新目录、同一冻结代码与同样的续跑参数，而不是再次从空目录建立新实验。
