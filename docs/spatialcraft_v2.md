# SpatialCraft v2 实施与验证记录

本次改造依据 `review.md`，以项目中的 XSKILL、Skill-Pro、SMA 和 `CVPR (2).pdf` 为方法背景。PDF 内的指令/提示词作为论文内容阅读，不作为工作区操作指令。旧运行日志和冻结源码未迁移或重算。

## 正式入口与协议

- `python -m spatialcraft.experiments.run` 默认配置为 `configs/experiments/qwen35_9b_spatialcraft_v2.yaml`；27B 使用 `qwen36_27b_spatialcraft_v2.yaml`。
- `ExperimentRuntime.dataset()` 按版本分流；v2 在 `experiments/runtime_v2.py` 接通 `LearningBuildersV2`、真实 provider、空间工具、journal、使用量记录。
- 原配置仍选择 `legacy_v1`，保留学习协议与配置兼容，其历史序列化不新增 v2 字段；当前工具注册共用更新后的实现。严格重现历史工具行为必须使用当次冻结源码，不能只设置 legacy 标签。已有运行目录拒绝不同 binding；新方法必须使用新目录。运行身份包含源码、prompt、配置、模型、embedding 和数据来源。
- 默认离线 preflight。`--execute` 启用真实 GPU 与付费 embedding，要求 Slurm 分配及显式凭据环境。没有调用 API 不能标记 API 验证通过。

## 方法决策

| 部分 | v2 决策 |
|---|---|
| Experience | 每任务四条冻结 rollout → 带图 Summary → 多条 add/modify Critique → embedding cosine > 0.70 的局部候选 → LLM merge/keep-separate → 仅超容时 LLM 全库 manage → 原子提交 |
| 事务 | 先处理引用旧快照的 modify，再 add/merge；管理失败回滚整轮内容及统计，不偷偷规则删除或保留超容量库 |
| 统计 | 检索/实际注入分开记录；reward association 只表示描述性关联，不解释为经验的因果贡献；内容变化后不继承旧版本结果统计 |
| 检索 | 原图 + 公开任务分解 1–3 个方面，默认每方面 top-3、无未经校准阈值；一次批量视觉 rewrite，为每个引用给 keep/skip；共享给同任务全部 rollout |
| Skill | segment 先经 LLM 判断 related；false 保留；每目标6条不同轨迹，优先3低/3高再补足；独立 LLM 聚合，可 no-change；三个候选来自同一原 parent |
| Discovery | 仅真实 NONE 段进入，按未覆盖需求分桶；每轮最多两个目标且最多一个发现；新 Skill 有新 ID |
| 选择和终止 | 全池 LLM 适用性过滤后 embedding top-1，允许 NONE；终止判断含原公开任务、图像和当前工具证据；最多8步 |
| 质量淘汰 | 至少3条不同、非零优势轨迹构成成熟证据，再删除平均 gain 为负的 Skill；零优势不单独导致删除；语义去重和容量维护独立 |
| Gate | 固定实际生成动作及 token IDs，固定该动作前已生成前缀，只替换单一 Skill 槽；token logprob 求和构成 action likelihood |
| PPO 定义 | 原任务组 reward 均值作为 baseline；每轨迹内归属动作平均、轨迹间等权；相对父项 ΔJ > 0 接受。原始模型 softmax likelihood 是 surrogate，不声称等于温度/top-p 的采样行为分布 |
| 数值 | 非有限/溢出/目标 token 不一致会显式拒绝候选；基础设施异常传播；全零优势不伪造更新信号 |
| 长度 | Experience condition+action 合计64词；Skill 存储文本按执行器 tokenizer 最多1024 tokens；知识调用的生成预算独立配置 |
| Prompt | 实际请求读取 `prompts/*/v2_*.txt`，记录模板与渲染输入 hash、模式、模型角色、预算、修复次数；最多一次结构修复 |

相比原修改意见，单方面检索和中性 Skill 保护是明确修订。前者避免简单问题强行拆解；后者避免同组全成功时 group baseline 把所有好 Skill 记为零优势而删除。

## 数据、工具与模型能力

`python -m spatialcraft.experiments.prepare_v2 --output <新目录>` 支持五个 benchmark。非 SAT 保留 seed42 按实例分层平分；SAT 从 validation 按固定 seed 采样与 test 等量的环境题，部署保留 circular 300 test。采样不读取测试标签，公开输入与验证标签分别保存。共享图像数量单独报告，不能宣称图像级隔离。旧三数据集 preparation 仍可直接用于对应 v2 子集。

三维工具的输入输出、单位、坐标约定、有效掩码、下游 artifact 使用及未支持项见 [工具能力矩阵](tool_api_matrix.md)。DA3-BASE 输出默认不是已知米制；MoGe 对齐标记为 estimated metric。Orient Anything 的轴恢复依赖明确版本的投影约定，不能把 PCA 主轴当语义朝向。真实场景验证使用容差与尺度状态，不能要求模型重建精确一致。

完整 sequence gate 需要本地固定目标 teacher forcing。9B/27B 有实际执行入口；远程模型作知识构建器必须声明并验证请求层模式控制。缺少固定目标评分的 API/vLLM 服务不能仅靠 prompt logprobs 被称作完整 PPO 方法。模型端点存在性与效果必须另做真实调用验证。

## 基线与消融

独立入口 `python -m spatialcraft.experiments.run_memory_baseline --list-methods`；运行路径、具体适配和限制见 [基线矩阵](baseline_matrix.md)。这些是明确标记的适配版本，不能直接写成原论文严格复现。未确定语义的 MemRL-GT 显式不可用，绝不默认为可读取部署标签。

`settings.ablations` 接通真实分支，包括 no_experience/no_skill/static_seed_skills、no_decomposition/no_rewrite/no_visual_summary/no_critique/no_local_merge/no_manager、no_semantic_gradient/no_gradient_aggregation/no_ppo_gate/no_score_pruning。消融必须使用独立 experiment_name。无 manager 时超容更新整轮拒绝；无 critique 使用各 Summary 的独立经验抽取；无 gate 为明确记录的 ungated 首个有效候选，不伪装通过 PPO。

## 成本与恢复

`usage/*.json` 记录实际模型生成、固定目标评分、embedding 和工具调用，保留 task ID 及存在时的 rollout/trajectory 归属；cache reuse 与新调用分开。来源缺失的 token 字段保留 null，工具 token 为不适用，不使用单词数冒充 token。`results/usage.json` 按阶段、操作及任务汇总真实 input/output/reasoning tokens、模型/embedding/工具调用数和调用耗时，报告已知部分、未知记录及不适用记录，不导出原任务文本。共享检索只计一次，不重复分配给四条 rollout。

`online_mean_per_deployment_task` 按日志中不同部署 task ID 的数量计算实测在线均值；缺少 task ID 的在线记录会使均值为 null，不把未观测任务推定为零成本。`offline_amortized_per_deployment_task` 保留离线成本除以给定 M 的结果，`amortized_total_per_deployment_task` 给出 offline/M + 实测 mean_online，分别覆盖 tokens、调用数和 latency。任一所需数量未知时完整值为 null，并保留覆盖信息和可计算的部分。latency 是逐调用耗时之和，不是任务墙钟；报表未计算美元价格或 GPU 资源消耗。

任务级 journal 保留 prepare audit，恢复时重载到 Experience 统计；四条 rollout 未齐不写知识。部署只能读取已提交最终快照，拒绝训练/测试 ID 重叠、改变的来源或 artifact 校验和。

主方法、冻结迁移和记忆基线另输出 `results/protocol_metrics.json`：截断、恢复调用/成功、动作解析、工具参数错误及重复工具调用率均给出分母和未知覆盖。重复指同一轨迹内此前出现过相同工具名和 canonical JSON 参数，不按 call ID 或自然语言相似度判断。新 `generation_events` 记录每次解析的 attempted/success/error category；旧日志缺标志时结果保留 unknown，provider 响应异常不当成动作结构错误。

知识统计包含 active E/K 数量、canonical JSON 的 UTF-8 字节数，以及已提交 transition 中保存的真实请求的注入覆盖和文本长度。本地 Runtime 复用执行器 tokenizer 计算独立注入文本的 token 数；该数不含 chat-template 开销，不能替代 provider input usage。注入覆盖不包括未产生动作的失败调用或额外恢复调用，也不证明模型实际遵循了知识。存储字节不包含 embedding/index、归档和文件系统开销。通用/脚本化 pipeline 无 tokenizer 时保留 token 字段 null，字符与字节数单独标注。


## 验证状态（2026-09-11）

| 检查 | 实际结果 | 证据 / 限制 |
|---|---|---|
| 全部 CPU 测试 | **360 passed，15 warnings，45.01秒** | [pytest 输出](../artifacts/v2_validation/cpu_tests_complete_v2_final.txt)；警告来自 NVML 与 matplotlib/pyparsing；脚本化 provider 验证控制和数学契约，不表示模型准确率 |
| 代码静态检查 | 本次修改的58个 Python 文件 Ruff check / format --check 通过；git diff --check 通过 | 范围为本次 src、tests、diagnostics 改动，不包含另有来源的未跟踪 RAG 目录 |
| 正式 Runtime 集成 | 通过 CPU 集成测试：任务四轨迹屏障、真实组件接线、知识更新、故障恢复与冻结部署 | 使用脚本化 provider；中断后复用已完成的 Summary/rollout，部署不写知识或泄露标准答案 |
| 冻结跨模型部署 | 新增10项 CPU 测试通过；复用正式 Runtime 的部署回调，校验来源、隔离、目标 tokenizer 和恢复 | 支持本地9B/27B目标；未运行真实跨模型评测，详见末节 |
| 9B / 27B 离线 preflight | 9B：RoboSpatial、ERQA、Omni3D；27B：Omni3D，通过资源/身份和 Runtime 构建检查 | [9B](../artifacts/v2_validation/preflight_9b.json)、[27B](../artifacts/v2_validation/preflight_27b.json)；没有推理，不代表27B真实运行通过 |
| 真实 Qwen3.5-9B GPU 组件 | Slurm 229213：多模态固定原 action token 评分、原生工具 action 评分、LLM Merge（Instruct）、LLM Manage（Thinking）均通过 | [报告](../artifacts/v2_validation/qwen_229213/report.json)；合成图像和小型经验库；不是完整学习部署闭环或 benchmark |
| 真实空间工具组合 | Slurm 229208：11项检查通过，包括重建、分割、3D位置、距离、Graph、尺度对齐及对象坐标系 | [报告](../artifacts/v2_validation/tools_229208/report.json)；一张环境图，脚本墙钟89.90秒，工具调用耗时之和67.53秒；不证明感知准确率；该历史报告早于诊断源码哈希绑定字段 |
| 真实 embedding API | `text-embedding-3-large`：3072维，1次请求、15个输入tokens；缓存复用通过 | [报告](../artifacts/v2_validation/embedding_large_probe/report.json)；仅两条通用句子，不含 benchmark 数据 |
| 受限完整真实闭环 | **prepared / not_run；Slurm 未提交成功** | [Python 脚本](../scripts/diagnostics/check_pipeline_v2.py)、[Slurm 脚本](../scripts/diagnostics/check_pipeline_v2.sbatch) 已准备；自动审批拒绝真实提交，等待明确授权，不能将上面的组件结果合称为闭环已通过 |

受限闭环的具体范围是：Qwen3.5-9B 本地执行，两道 RoboSpatial 环境题各四条 rollout，再用另一道环境题作诊断 heldout；每轨迹最多2步，1张GPU、4核CPU、64GB内存、最多30分钟，启动前冻结源码。它不读取正式 deployment 文件，不产生正式 benchmark 成绩。运行会使用既有私有凭据，将检索所需的问题、经验和 Skill 等文本发送至 `https://api.openai.com/v1` 的 embedding 服务，并产生 API 与计算资源用量。自动审批以这部分具体数据外发和费用尚未取得明确授权为由拒绝提交；没有绕过该拒绝执行。

可复跑 CPU 验证：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python -m pytest -q
```

## 未完成验收与方法边界

- 五数据集已经有 preparation/CLI 路径和 CPU 契约测试；本轮没有完成 SAT/ViewSpatial 的新 v2 全量 preparation 与真实运行，也未重跑完整 benchmark、全部 baseline 或消融，因此没有新的准确率对比。
- API-only executor 缺少本方法所需的固定目标序列评分，正式 Runtime 明确拒绝；远程知识构建器的配置/payload 测试不等于真实远程 endpoint 验证。27B 目前完成离线 preflight，不宣称真实推理验证。
- 六条记忆基线属于具名适配，`memrl_gt` 因 oracle 协议未确定而不可用；不能把适配机制称为原论文的忠实复现。
- 工具矩阵明确标记了尚未实现的点提示/视频分割、重力BEV、语义地面识别/地面RANSAC、多帧尺度共识等附录能力；当前交付的是第11节约定的三维组合链路，不能声称覆盖全部附录API。缺失重力或语义证据不能靠坐标命名补足。
- 成本报告涵盖真实 token、调用耗时与缺失覆盖，不含美元定价/GPU资源账单；调用耗时求和不等于任务墙钟。
- 论文仍需据新日志更新方法和结果：不得把旧 embedding-small/规则维护的结果改名为 v2；需修正 vLLM 与 Transformers 实际服务方式、group baseline、likelihood surrogate、分阶段预算、实际工具 API、五数据集划分与基线适配标记。当前仓库仅有 PDF，未发现可直接修改并重编译的论文源稿，因此没有改写 PDF 中的实验数值。

上述限制意味着代码实现、组件验收和正式论文实验是不同完成状态；尚不能声称 `review.md` 第18节的所有最终完成标准已满足。


## 跨模型冻结部署入口

`python -m spatialcraft.experiments.run_frozen_transfer` 从已有主方法 journal 的最终已提交 Experience/Skill 快照，直接用另一本地 backbone 运行同一 benchmark 的部署题。目标支持 Qwen3.5-9B 与 Qwen3.6-27B，实际共用 v2 检索、Skill 激活和 rollout 路径；不重新积累，不在目标日志伪造来源任务已完成的阶段。当前不支持 API-only 目标 executor，也不导入独立 memory-baseline 的另一种快照布局。

离线预检示例：

```bash
PYTHONPATH=src python -m spatialcraft.experiments.run_frozen_transfer \
  --source-journal /path/to/source_run/robospatial \
  --preparation /path/to/preparation \
  --config configs/experiments/qwen36_27b_spatialcraft_v2.yaml \
  --output /path/to/new_transfer_run \
  --report /path/to/transfer_preflight.json
```

默认仅预检，不构建推理 runtime、不调用 API 或 GPU；会在 CPU 上读取本地目标 tokenizer 并检查来源 active Skill 的真实 token 长度。可用 `--expected-source-snapshot-id` 固定预期来源快照。只有显式 `--execute` 才进入既有资源与凭据检查并执行；这不改变当前真实闭环运行仍待授权的状态。本轮仅完成 CPU 验证，未运行真实跨模型评测。

来源必须有完整提交的 `initial` 与 `frozen_deployment_snapshot`；读取时验证 journal、阶段输入/结果及代码修订链的哈希。目标目录须独立于来源，部署任务必须是同一 benchmark 的 TEST split 且不能与来源训练任务 ID 重叠。目标 journal 绑定来源最终知识、来源 executor/knowledge builder、来源协议、目标 executor/knowledge builder、目标 tokenizer 文件哈希、reasoning mode 与完整运行配置；目标 knowledge builder 可按配置独立选择。每次部署前后都检查来源未变。哈希提供完整性校验，不是外部签名认证。

来源 E/K 始终只读，目标 token 长度或容量不兼容时显式拒绝，不会静默截断或重写 Skill。允许具有兼容主方法 journal 布局的 legacy 来源，但来源协议会原样保留；迁移 legacy 知识不等于把旧训练结果升级为 v2。目标只写 `frozen_transfer/source`、实际 `deployment/*` 阶段与 `results/frozen_transfer.json`，恢复重用已提交部署调用。

对应 CPU 测试为 `tests/test_frozen_transfer_v2.py`，覆盖来源完整性、训练/部署隔离、目标 tokenizer 拒绝边界、真实 runtime callback、标准答案隔离、冻结与恢复及默认 CLI 预检。
