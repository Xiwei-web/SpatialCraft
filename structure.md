# SpatialCraft 快速结构说明

> 主方法版本（2026-09-11）：默认入口已升级为 `spatialcraft_v2`，结构增量见本文第5节，配置、验收与限制见 [v2实施记录](docs/spatialcraft_v2.md)。以下带日期的旧运行状态是历史记录；RAG等独立baseline各自保留其运行协议。

本文用于让新对话中的 Agent 快速了解 SpatialCraft 的代码、空间工具和数据集位置。

GPT-5.4 ViewSpatial RAG变更次数续跑（2026-09-12）：已提交 **233606**，继承229405的375条已完成环境输出（前93题各4次、第94题3次），剩余2762题各1次，最终计划3137条环境记录；测试2856题各1次。medium/16384、text-embedding-3-large/Top-3不变。新增`RAG/continuation.py`及`gpt54_viewspatial_remaining1_medium16384.sbatch`，严格校验原输入和结果后导入新journal。实际冻结版本40项离线回归通过；提交后RUNNING / cn-06，源码校验已通过，真实数据检查/导入尚待运行完成。共享存储读取阻塞期间，正式输出改放`/home/xiwei.liu/spatialcraftRuns/gpt54_rag_viewspatial_remaining1_medium16384_20260912_v1`，源码放共享home的`spatialcraftSnapshots/`；同名`/l/users/...`目录仅是未提交的准备目录。详见 [233606续跑说明](RAG/runs/20260912_gpt54_viewspatial_remaining1.md)。

GPT-5.4 ViewSpatial RAG（2026-09-11）：已提交 **229405**，gpt-5.4 / medium / 单次16384 tokens，环境2856题×4次、测试2856题×1次，text-embedding-3-large / Top-3原始样例检索，独立从空库建库。与229312冻结代码和输入一致，输入/媒体校验通过，复用229347的GPT-5.4真实API验收。提交时 **PENDING / QOSMaxJobsPerUserLimit**，已有3项RAG任务运行，名额释放后由Slurm自动调度。独立运行 `spatialcraftLog/runs/gpt54_rag_viewspatial_medium16384_20260911_v1`；启动后日志 `slurm-229405.out`，最终accuracy `viewspatial/results/deployment.json`。详见 [GPT-5.4 ViewSpatial RAG运行说明](RAG/runs/20260911_viewspatial_gpt54_medium16384.md)。

GPT-5.4 RAG（2026-09-11）：已提交 **229347**，与两个mini任务独立并行；gpt-5.4 / medium / 单次16384 tokens，环境每题4次、部署每题1次，Top-3原始样例检索。RoboSpatial、ERQA、Omni3D、SAT测试175/200/250/300题，执行源码及数据输入与229287一致，GPT-5.4独立从空库建库；真实GPT-5.4建库→embedding检索→多图部署API验收通过。运行 `spatialcraftLog/runs/gpt54_rag_medium16384_20260911_v1`；日志 `slurm-229347.out`，最终 `<dataset>/results/deployment.json` 及 `results/accuracy.md`。提交时PENDING，状态以Slurm为准。详见 [GPT-5.4 RAG运行说明](RAG/runs/20260911_gpt54_medium16384.md)。

ViewSpatial RAG（2026-09-11）：已提交 **229312**，与229287独立并行；gpt-5.4-mini / medium / 单次16384 tokens，环境2856题×4次、测试2856题×1次，Top-3原始样例检索。复用229287的同一冻结代码，ViewSpatial输入与此前mini直接回答baseline逐字节一致；完整划分/媒体离线检查通过，无跨集合内容完全重复。独立运行 `spatialcraftLog/runs/gpt54mini_rag_viewspatial_medium16384_20260911_v1`；实时日志 `slurm-229312.out`，进度 `viewspatial/progress.json`，最终accuracy `viewspatial/results/deployment.json`。提交时PENDING，状态以Slurm为准。详见 [ViewSpatial RAG运行说明](RAG/runs/20260911_viewspatial_gpt54mini_medium16384.md)。

RAG正式运行（2026-09-11）：已提交 **229287**，gpt-5.4-mini / reasoning_effort=medium / 单次16384 tokens，环境每题4次、部署每题1次，Top-3原始样例检索；不传temperature/top_p/seed。RoboSpatial、ERQA、Omni3D、SAT部署175/200/250/300题，与先前baseline输入逐字节一致。已处理RoboSpatial和Omni3D各一组跨集合内容重复：保持原划分，检索时排除当前题的完全重复历史样例并记日志。34项回归、冻结预检查及独立真实API建库→检索→多图部署验收通过。运行目录 `spatialcraftLog/runs/gpt54mini_rag_medium16384_20260911_v2`；日志 `slurm-229287.out/.err`，最终 `<dataset>/results/deployment.json` 和 `results/accuracy.md`。提交时PENDING，实时状态以Slurm为准。详见 [RAG运行说明](RAG/runs/20260911_gpt54mini_medium16384.md)。

RAG 示例检索 baseline（2026-09-11）：新增独立 `RAG/`，支持 gpt-5.4-mini 和 gpt-5.4。环境阶段保存公开任务与模型原始输出，部署按任务 embedding cosine Top-3 检索历史样例；不生成反思总结、lessons、Experience 或 Skill。默认单次视觉问答、reasoning=none、4096 tokens，环境每题4次、部署每题1次；两个模型各自建库。**该段为最初编写阶段记录；随后提交的medium/16384任务见上方运行说明。** 详见第6节及 `RAG/README.md`。

Tool-only ReAct baseline（2026-09-10）：已提交 **227082**，gpt-5.4-mini / reasoning=none / temperature=0 / 每次4096 tokens。复用纯模型baseline同一925题（RoboSpatial175、ERQA200、Omni3D250、SAT300），每题一条多步trajectory、单pass，使用项目11个真实空间工具，无Experience/Skill/embedding/PPO。50次工具交互上限、单step最多一次1024-token恢复、步数耗尽512-token最终作答。单张A100 40GB，四数据集依次执行。运行 `spatialcraftLog/runs/gpt54mini_tool_react_baseline_20260910_v1`；最终 `{dataset}/results/deployment.json`、汇总 `results/summary.json`、实时 `{dataset}/progress.json`。202项完整回归、最终63项针对性回归、真实API工具往返及六个重型GPU工具验收通过。详见 `docs/gpt54mini_tool_react_baseline_20260910.md`；实时状态以Slurm为准。

ViewSpatial GPT-5.4 baseline（2026-09-10）：已提交 **227000**，仅复用固定deployment的 **2856题**，gpt-5.4 / reasoning=none / temperature=0 / 4096 tokens / 单轮每题一次，无工具/E/K。输入与mini任务226866逐字节一致，seed42的既有50/50划分保持不变。独立运行 `spatialcraftLog/runs/gpt54_viewspatial_deployment2856_seed42_20260910_v1`；最终accuracy见 `viewspatial/results/deployment.json`，实时见 `viewspatial/progress.json`。启动配置和冻结输入检查通过，详见 `docs/viewspatial_gpt54_deployment_baseline_20260910.md`；状态以Slurm为准。

ViewSpatial 半集 baseline（2026-09-10）：已提交 **226866**，gpt-5.4-mini / reasoning=none / temperature=0 / 4096 tokens / 单轮每题一次，无工具/E/K。按question_type分类内50/50、seed42、奇数余项environment优先交替分配，5712题得到environment/deployment各2856题；**本次只评测deployment的2856题**。共享划分 `spatialcraftLog/preparation/viewspatial_seed42_v1`；运行 `spatialcraftLog/runs/gpt54mini_viewspatial_deployment2856_seed42_20260910_v1`，结果 `viewspatial/results/deployment.json`。48项回归、全部请求校验、32图API验收通过。5712/2856为题目记录数，多图题保留全部视图。详见 `docs/viewspatial_mini_deployment_baseline_20260910.md`，实时状态以Slurm为准。

GPT-5.4 baseline（2026-09-10）：已提交三轮 **226631/226632/226633**，自动汇总 **226634** 依赖三轮成功。gpt-5.4 / reasoning=none / temperature=0 / 4096 tokens / 每题一次；每轮同一925题（RoboSpatial175、ERQA200、Omni3D250、SAT300），无工具/E/K，合计2775次。独立目录 `spatialcraftLog/runs/gpt54_direct_baseline_20260910_three_runs_v1`，单轮 `runN/{dataset}/results/deployment.json`，三轮 `results/accuracy_mean_std.{json,csv,md}`（样本std，ddof=1，另存variance）。33项回归、冻结preflight及真实gpt-5.4多图API验收通过。详见 `docs/gpt54_three_runs_20260910.md`；状态以Slurm/文件为准。

GPT baseline 三轮评测（2026-09-10）：首轮 **226209已完成**，RoboSpatial/ERQA/Omni3D/SAT准确率为52.57%/34.50%/22.40%/57.67%。新复跑 **226421（rep43）**、**226422（rep44）**，自动汇总 **226423** 依赖两者成功；同一925题、首轮冻结代码、reasoning=none、temperature=0、4096 tokens。**Responses API不支持生成seed，43/44仅作独立复跑标识，不声称控制了模型随机种子。** 结果在 `spatialcraftLog/runs/gpt54mini_direct_baseline_20260910_repeats_v1/results/accuracy_mean_std.{json,csv,md}`，std用样本标准差ddof=1，另存variance。详见 `docs/gpt54mini_three_runs_20260910.md`。后续状态以Slurm/结果文件为准；下文首轮运行中说明为历史记录。

纯 GPT baseline（2026-09-10）：已提交 **226209**，启动于 CPU 节点 cn-03。gpt-5.4-mini / reasoning=none / 4096 tokens / 每题一次，无工具、Experience、Skill；RoboSpatial 175、ERQA 200、Omni3D 250、SAT 300，共925题。独立运行目录 `spatialcraftLog/runs/gpt54mini_direct_baseline_20260910_v1`；实时 `{dataset}/progress.json`，完整结果 `{dataset}/results/deployment.json` 与 `results/summary.json`。8项关键回归、全部请求检查及真实16图API验收通过。详见 `docs/gpt54mini_baseline_20260910.md`。当前状态以 Slurm/结果文件为准。

源码策略更新（2026-09-09）：已实现用户要求的 **action_recovery_v2**。4096-token生成截断先检查完整action，无合法action时同step最多一次1024-token恢复，保留工具；只有50次工具交互耗尽后才追加一次512-token最终作答。Recovery不占工具步，失败记为明确模型失败；日志分别统计正常/恢复/最终调用及截断事件。153项完整回归和Omni3D离线初始化通过。正式/通用loop共用实现，PPO记录实际请求。**已提交冻结作业仍执行旧快照；新策略须独立运行，不能混用旧协议结果。** 详见 `docs/action_recovery_v2.md`。

最新续跑（2026-09-09）：**220852 已提交，查询时 PENDING/Priority**，使用 `code_revisions/output_budget_v1` 和新入口 `scripts/inference/robospatial_budget.sbatch`。2×A100 40GB + FLA、96GB主机内存、48小时；115项实际快照适用回归通过。保留372条训练轨迹/92次Experience更新，从第93题Experience更新继续，采用4096-token超限后单次强制收尾策略。日志仍在原RoboSpatial运行目录，详细说明见 `docs/output_budget_recovery.md`。以下未重提/未切换说明为历史状态。

输出超限策略更新（2026-09-09）：项目源码已加入 `force_completion_v1`：4096-token 超限后追加一次强制收尾，仍失败则标记并继续，不写入截断知识。详见 `docs/output_budget_recovery.md`。**本次未重提作业，旧冻结快照尚未切换**。220740 双 GPU/FLA 验证已通过；220741 在第 93 题 Experience 摘要输出超限退出，保留 372 条训练轨迹、92 次 Experience 更新，尚无最终测试成绩。以下排队状态为历史记录。

最新状态（2026-09-09）：按用户要求准备 **FLA高效线性注意力 + 2×A100 40GB**。依赖隔离在 `/l/users/xiwei.liu/tool/env_overlays/qwen_fla_052`，不改共享conda；候选快照为原运行目录下 `code_revisions/fla_dual_v2`。110项完整回归及43项冻结快照关键测试通过；**双GPU实测尚未通过，作业220740正在等待资源**。正式续跑220741依赖其成功，失败则不启动。原368条轨迹/92次Experience更新保留，尚无完整测试成绩。最新启动、验证和断点说明：`docs/fla_dualgpu_20260909.md`；此前状态均为历史记录。

Omni3D journal修复（2026-09-09）：**220630** 的GPU和27B推理验收通过，正式实验在保存整数GPU编号与字符串cpu混合键时发生JSON排序错误，尚无训练轨迹。已修复记录序列化，并将真实pipeline/journal创建、写入、重开加入离线检查，107项测试通过。新任务 **220679 已提交**：Qwen3.6-27B / thinking=false / 4096 tokens / 每题4 rollouts / 单pass，**1×A100 40GB + 128GB主机内存**，保留BF16+CPU卸载及上次卷积修复。独立运行目录与监控见 `docs/omni3d_27b_run.md`，本次说明见 `docs/omni3d_27b_journal_fix.md`。作业状态以Slurm为准。

最新资源重提（2026-09-08）：217328在368条训练轨迹/92次Experience更新后，于第93题第1条rollout的第40步发生CUDA OOM。按用户要求提交 **218286**，仍用当前冻结快照 `json_output_v1` 和单张完整 **A100 40GB**，CPU内存64GB；启用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 缓解碎片，未改实验超参数。集群可见GPU均为40GB单卡，每节点4卡合计160GB，不会自动合并。新作业状态以Slurm为准；历史状态如下。

实时状态补充（2026-09-08 17:26）：作业 **217328 已在 gpu-04 正式续跑**，原JSON故障点已通过，后续演化模型调用持续落盘；恢复后再次校验28,371份旧JSON均未改写。完整测试尚未完成。

当前实验（2026-09-08 17:23）：仅 RoboSpatial，Qwen3.5-9B **instruct / thinking=false**，单次输出与 Skill 生成均为 **4096 tokens**。检测框修复后作业215987已完成220条训练轨迹、55次Experience更新，随后在Skill语义梯度JSON解析处退出。本次受限语法修复通过实际响应回放及98项测试；新续跑作业 **217328** 已提交，初始排队，尚无完整成绩。目录：`/l/users/xiwei.liu/spatialcraftLog/runs/robospatial_qwen35_9b_instruct4096_v1`。入口 `scripts/inference/run_robospatial.sh` 根据 `robospatial/code_patch.json` 自动使用 `code_revisions/json_output_v1`，保留检测框修复；所有旧快照和已提交结果不改写。最新恢复说明：`docs/json_recovery_20260908.md`。作业状态以Slurm查询为准。

## 1. 项目代码：`/home/xiwei.liu/spatialcraft`

核心代码位于 `src/spatialcraft/`：

- `schemas/`：统一数据结构，包括 `TaskSample`、Action、ToolResult、SpatialState、Trajectory、Experience、Skill 和 KnowledgeSnapshot。
- `storage/`：原子文件写入、目录布局、manifest、checkpoint 和知识快照存储。
- `models/`：统一模型接口及 OpenAI、Gemini、OpenAI-compatible、vLLM、本地 Transformers provider；`models/scoring/` 实现固定 Action 的严格 token log-prob 评分。
- `datasets/`：RoboSpatial、ERQA、Omni3D、SAT、ViewSpatial 适配器，将原始样本统一转换为 `TaskSample`。
- `verification/`：精确匹配、选择题、数值、空间关系及 LLM Judge，将 Agent 输出统一转换为 reward。
- `tools/`：工具契约、注册与执行、Artifact Store、坐标系、mock 工具和真实空间工具适配器。
- `agent/` 与 `rollout/`：多步 MLLM→工具→Observation→答案执行循环、Skill 生命周期、轨迹记录和批量 rollout。
- `knowledge/experience/`：Experience 检索、改写、视觉总结、跨 rollout 批判、更新和合并。
- `knowledge/skill/`：种子 Skill、选择与生命周期、credit、语义梯度、候选生成、lineage、剪枝和严格 NP-PPO Gate。
- `knowledge/coordinator.py` 与 `pipelines/`：冻结 batch barrier、Experience/Skill 联合更新、原子 snapshot 提交和只读部署。
- `evaluation/`：accuracy、pass@k、工具/Experience/Skill/效率/迁移指标，以及八类 baseline 和消融配置。
- `experiments/`：三个 benchmark 的固定分层划分、公开输入/私有标签、完整 Qwen9 协议运行器和逐阶段断点日志；`python -m spatialcraft.experiments.run` 默认只做离线检查，`--execute` 才启用真实推理及付费 embedding。当前协议见 `docs/confirmed_protocol.md`。
- `resources/skills/seed_skills.yaml`：六个初始程序化技能。

其他重要目录：

- `RAG/`：独立 GPT 示例检索 baseline；保存原始 task/output，经冻结任务向量索引检索，不调用 Experience/Skill 学习分支。入口为 `run_gpt54mini.sh`、`run_gpt54.sh`，详见第6节。 `continuation.py`支持在新目录导入已完成前缀并减少剩余环境题的rollout次数；`tests/test_continuation.py`验证导入和恢复。
- `configs/models/`：五个 backbone 配置，以及 `text-embedding-3-small.yaml`；Experience/Skill 共用 OpenAI embedding，不回退到 Qwen embedding。
- `configs/experiments/qwen35_9b_spatialcraft.yaml`：1 pass、每题 4 rollouts；每 parent 6 条相关 trajectory 后演化、每 task 屏障最多 2 parents；trajectory/Skill 上限50/8 steps、单次/候选输出4096 tokens、thinking=false（instruct）；PPO直接评分action tokens；top3、候选3、容量100/20。
- 补充确认：训练 top_p=0.9、部署/辅助构建 temperature=0、PPO epsilon=0.2 与严格正收益 margin=0、旧版本剩余轨迹仅存档；当前仅 `image_max_pixels=1048576` 仍待确认。
- `configs/roles/`：executor、knowledge builder、verifier、embedding、PPO scorer 角色配置。
- `prompts/experience/`：Experience 分支提示词。
- `scripts/data/`：benchmark 下载、检查和 SAT 300 条扩展脚本。
- `tests/`：阶段 3–12、已确认协议、embedding、完整运行器接线与断点/冻结部署，以及 v2 的 per-parent FIFO 队列、fixed-thinking 评分、50/8-step 测试。最新结果见 `spatialcraftLog/validation/protocol-v2-20260907/`；单元测试不等同于真实全链路已通过。
- `STATUS.md`：项目阶段完成状态。
- 根目录 PDF：SpatialCraft 草稿及 XSkill、Skill-Pro、SMA 参考论文。

最小使用环境：

```bash
conda activate spatialcraft
cd /home/xiwei.liu/spatialcraft
export PYTHONPATH=/home/xiwei.liu/spatialcraft/src
pytest -q
```

## 2. 空间工具：`/l/users/xiwei.liu/tool`

该目录约 28 GB，保存空间工具源码、模型权重和下载缓存：

- `repos/`：GroundingDINO、SAM3、MoGe、Depth Anything 3、Orient Anything、EasyOCR 源码；也包含 XSkill 和 Skill-Pro 参考仓库。
- `checkpoints/`：上述模型的本地权重，包括 GroundingDINO、SAM3、MoGe-2、DA3、OriNet/DINOv2 和 EasyOCR。
- `algorithmic/`：无需大型模型的 Geometry、Mask、Graph、Draw、Farneback Motion 等算法工具目录。
- `cache/`：Hugging Face、pip、Torch 和 EasyOCR 下载缓存；通常不应作为代码入口。
- `manifests/`：工具准备状态或校验清单存放位置。

SpatialCraft 当前注册的真实工具为：`detect`、`segment`、`mask`、`geometry`、`scale`、`reconstruct`、`pose`、`graph`、`motion`、`ocr`、`draw`。模型采用 lazy loading，创建 registry 时不会立即占用 GPU，首次调用对应工具时才加载权重。

```python
from spatialcraft.tools.real import create_real_tool_registry

tools = create_real_tool_registry()
print(tools.names())
```

代码默认使用 `/l/users/xiwei.liu/tool`；迁移目录时可设置：

```bash
export SPATIALCRAFT_TOOL_ROOT=/new/tool/path
```

## 3. Benchmark：`/l/users/xiwei.liu/benchmark`

该目录约 4.9 GB，保存五个空间推理数据集及本地媒体缓存：

- `RoboSpatial/`：`context`、`configuration`、`compatibility` 三类 Parquet；请求 `test` 时适配器会合并三类。
- `ERQA/`：官方 test Parquet。
- `Omni3D/`：Omni3D benchmark Parquet 和原始 `omni3d-bench.zip`；当前适配器统一暴露为 `test`。
- `SAT/`：train、static、validation、test；默认 test 使用 `SAT_test_circular_300.parquet`。同时保留原始 150 条和其他备份文件。
- `ViewSpatial/`：`ViewSpatial-Bench.json`，以及 `val2017.zip`、`scannetv2_val.zip` 图像资源。
- 各数据集下的 `.spatialcraft/media/`：适配器解包或物化后的图像缓存；`.cache/` 是下载缓存。

统一读取方式：

```python
from spatialcraft.datasets import create_default_registry

registry = create_default_registry()  # 默认根目录即 /l/users/xiwei.liu/benchmark
adapter = registry.create("sat")       # robospatial/erqa/omni3d/sat/viewspatial
samples = adapter.load_split("test", limit=10)
assert all(sample.dataset == "sat" for sample in samples)
```

如迁移 benchmark，可设置 `SPATIALCRAFT_BENCHMARK_ROOT`，或调用 `create_default_registry("/new/benchmark/path")`。后续 Agent 应通过这些适配器读取数据，不要直接假设不同原始数据集拥有相同字段。

## 4. Backbone 权重：`/l/users/xiwei.liu/model`

2026-09-07 下载的官方 post-trained 多模态 BF16 权重（不是 `-Base` 或量化变体）：

| 模型 | 本地目录 | 官方仓库 | 文件总大小 |
|---|---|---|---|
| Qwen3.6-27B | `/l/users/xiwei.liu/model/Qwen3.6-27B` | `Qwen/Qwen3.6-27B` | 55.59 GB，15 个权重分片 |
| Qwen3.5-9B | `/l/users/xiwei.liu/model/Qwen3.5-9B` | `Qwen/Qwen3.5-9B` | 19.33 GB，4 个权重分片 |

每个目录包含 safetensors 权重、分片索引、模型配置、tokenizer、chat template、图像/视频 processor 配置、官方 README 与许可证。固定版本及续传命令见 `/l/users/xiwei.liu/model/README.md`。`.cache/huggingface/` 是下载元数据，不是模型入口。

两个模型 YAML 的默认路径已对齐这里的 `model/`（不是 `models/`）。可显式覆盖：

```bash
export QWEN36_27B_PATH=/l/users/xiwei.liu/model/Qwen3.6-27B
export QWEN35_9B_PATH=/l/users/xiwei.liu/model/Qwen3.5-9B
```

2026-09-07 已升级当前 `spatialcraft` 环境：PyTorch 2.10.0 / CUDA 12.8、Transformers 5.5.4、vLLM 0.19.1、Accelerate 1.12.0。两个模型的配置与视觉预处理检查均通过；9B 已在分配的 A100 40GB 上分别通过 Transformers 和 vLLM 的真实 BF16 图片问答。27B 尚未验证完整 GPU 推理，需至少双 40GB 或单 80GB 卡并为 KV cache 留余量。本次没有保留后台模型服务。

依赖版本见 `requirements/inference-cu128.txt`；使用和验证记录见 `docs/inference_environment.md`。`scripts/inference/smoke_qwen.py` 提供离线及 GPU 验证，`scripts/inference/serve_qwen.sh` 提供仅监听本机的 Slurm 内 vLLM 启动方式。当前 27B 配置选择 `vllm_client`，需要先启动服务并设置 `QWEN36_27B_BASE_URL`；9B 配置仍选择 `transformers_local`。环境级验证不等于项目 provider 的多模态/工具调用/PPO 链路已全部通过验证。

升级前完整环境备份：`/l/users/xiwei.liu/env_backups/spatialcraft-before-qwen-20260907`，可按路径激活。现有 Python/Notebook 进程需重启以加载新版依赖。`pip check` 仅剩升级前已有的 TextWorld 平台提示。

## 5. 实验日志：`/l/users/xiwei.liu/spatialcraftLog`

Qwen3.5-9B 的 RoboSpatial → ERQA → Omni3D 实验准备说明见 `docs/qwen35_three_benchmarks.md`。
`preparation/qwen35_9b_seed42_v1/` 保存 175/175、200/200、251/250 的固定划分、评估专用标签和来源哈希；
`smoke/qwen35_9b_provider_v2/` 保存真实 provider 的少量调用及断点验证记录。
**目前没有正式三数据集 SpatialCraft 分数，也没有生成的实验 skill/experience。**
本轮已接通知识 builder、多模态 NP-PPO、工具反馈和严格协议运行器；新 thinking 协议验证见 `validation/protocol-v2-20260907/`，旧 nonthinking 验证保留在 `validation/protocol-20260907/`。
待用户设置 key 后再做真实 embedding/空间工具组合的小规模验收；未宣称全量实验通过。正式执行将写入 `runs/qwen35_9b_protocol_v2/{robospatial,erqa,omni3d}/`，共享 embedding 缓存位于该 run 根目录的 `embedding_cache/`；各 benchmark 的 `skills/round-*.json` 和 `checkpoints/pending_evolution.json` 保存演化队列及消费审计。
恢复使用同一个 `experiments.run --execute --output ...` 命令；变更代码/配置/权重/依赖需新版本目录。协议、暂定参数和完整日志索引见 `docs/confirmed_protocol.md`。


## 2026-09-11：SpatialCraft v2 正式入口

当前主方法默认配置为 `configs/experiments/qwen35_9b_spatialcraft_v2.yaml`，27B 对应 `qwen36_27b_spatialcraft_v2.yaml`。旧配置保留 legacy 学习协议，历史复现实验仍需使用当次冻结源码。

- `experiments/runtime_v2.py`、`learning_v2.py`：正式运行时与双层记忆集成；`operation_profiles.py`、`knowledge_generator.py`：按操作预算/模式、真实模板、模型角色与一次结构修复。
- `knowledge/experience/learning_v2.py`：视觉 Summary/Critique、embedding+LLM Merge、LLM Manage、检索重写与原子回滚。
- `knowledge/skill/learning_v2.py`、`RelatedEvolutionQueue`、`SequenceLikelihoodGate`：语义相关性、3/3六轨迹、聚合、REFINE/NONE发现、固定历史动作门控。
- `agent/action_target.py`、`models/providers/transformers_local.py`：真实动作token跨度和多模态固定目标评分。
- `experiments/usage.py`、`evaluation/cost_report_v2.py`：实际调用、未知usage、缓存与阶段成本。
- `experiments/prepare_v2.py`：五数据集准备；`run_memory_baseline.py`、`baseline_memory.py`：有独立执行分支的基线适配。
- `tools/spatial_arrays.py`、`tools/real/*`：可消费的三维点图、坐标/尺度/有效性契约及工具链接口。

新增 `evaluation/protocol_metrics_v2.py` 从真实轨迹审计生成恢复、解析、截断、重复调用率及知识存储/注入统计；`cost_report_v2.py` 提供按任务成本与离线摊销。

方法选择、验证范围、运行方式与限制以 `docs/spatialcraft_v2.md`、`docs/tool_api_matrix.md`、`docs/model_capabilities_v2.md`、`docs/baseline_matrix.md` 为准。GPU诊断与CPU检查输出位于 `artifacts/v2_validation/`，不属于正式benchmark结果。


### 跨模型冻结部署补充（2026-09-11）

- `experiments/run_frozen_transfer.py`：从已提交的主方法最终 E/K 快照直接运行另一本地 Qwen9B/27B 的 v2 部署；独立目标 journal 绑定来源与目标身份，不重跑积累、不改写来源。默认 CPU/offline preflight；API-only 目标不支持。
- `tests/test_frozen_transfer_v2.py`：来源哈希、split/task 隔离、目标 tokenizer 限制、正式 Runtime callback 和恢复冻结测试。入口与限制见 [v2 跨模型说明](docs/spatialcraft_v2.md#跨模型冻结部署入口)。

## 6. RAG baseline：`/home/xiwei.liu/spatialcraft/RAG`

这是用户定义的 example-retrieval baseline：环境阶段收集轻量原始 rollout 记录，部署阶段只读检索并提供历史任务文本和模型原始输出；不产生 reflected summaries 或 lessons。与 `experiments/baseline_memory.py` 中“仅保存验证成功工具轨迹”的 `rag_demonstrations` 变体不同。

- `run_rag.py`：共用入口，支持 `--model gpt-5.4-mini/gpt-5.4`、`--stage environment/deployment/all`、`--datasets`；默认仅离线准备，显式 `--execute` 才调用 API。
- `run_gpt54mini.sh`、`run_gpt54.sh`：模型便捷入口，需提供独立 `--output`；不会自动提交 Slurm。
- `data.py`：复用已有 deployment 输入，准备去标签的环境任务，校验图像、来源及跨集合任务重复。
- `core.py`：原始 task/output 记录、任务 embedding 文本、cosine Top-k 和样例注入，无 LLM 反思或改写。
- `runner.py`：环境收集、索引冻结、部署检索、确定性评分和逐调用断点日志。
- `tests/test_rag.py`：已编写的离线测试，覆盖标签隔离、错误答案保留、检索、冻结和恢复；本次未运行。
- `README.md`：完整方法、参数、划分、命令及结果路径说明。

默认两阶段 reasoning=none、4096 tokens；环境单pass每题4次、temperature=0.7；部署每题1次、temperature=0。默认 text-embedding-3-large，对 question、choices、answer_type 编码，Top-3 检索同数据集的不同历史任务。每个历史任务以最早完成且非空的原始输出作为代表，不按正确性筛选；全部 rollout 均存档。当前题传入全部原图，历史样例只提供文本。环境 rollout 数、温度、Top-k 和 embedding 型号可配置，变更须用新输出目录。

默认 RoboSpatial 175/175、ERQA 200/200、Omni3D 251/250、SAT 300/300（environment/deployment）；前三者复用既有划分，SAT 环境集采用现有 v2 validation→test 抽样规则，部署仍为 circular test 300题。可选 ViewSpatial 2856/2856，复用已划分半集。

运行后，原始记录在 `<output>/<dataset>/memory/records.jsonl`，冻结索引与标记为 `memory/index.json`、`memory/snapshot.json`；进度为 `<dataset>/progress.json`，逐调用日志在 `<dataset>/stages/`。accuracy 在 `<dataset>/results/deployment.json`，逐题结果在 `results/predictions.jsonl`，跨数据集总表在 `<output>/results/accuracy.md`。正式RAG运行已提交229287，目录和检查记录见上方运行说明；尚无完整accuracy结果。
