**GPT-5.4-mini / RoboSpatial / 完整 SpatialCraft：执行前检查报告**

检查日期：2026-09-14。代码基线：`b69d7d7`，同时检查了当前工作区文件。用户最终决定：保持 GPT-5.4-mini 与完整 PPO 要求，先完成报告，暂不提交。

**结论：当前不能直接提交一个满足上述要求的有效实验。** 普通 GPT 生成可以配置为 `reasoning_effort=medium`、`max_output_tokens=16384`；阻塞来自完整方法所需的固定目标动作概率重评分，以及主运行器对本地模型、Instruct 模式的约束。修改模型名或增加 token 上限不能解决这些问题。

本次只进行了源码检查、官方接口文档核对和 8 项离线探针；没有调用付费生成/embedding API、加载模型、运行数据集或提交 Slurm 任务，没有修改算法、运行代码或实验配置。新增内容仅为本报告及[离线检查证据](../artifacts/gpt54mini_spatialcraft_check_20260914/offline_checks.json)。检查证据包含源码 SHA-256、完整 Git HEAD、探针结果及时间。

**本次需求及前提检查**

| 项目 | 本次目标 / 解释 |
|---|---|
| 方法 | 完整 SpatialCraft，包括 Experience 学习、Skill 演化及严格 PPO Gate；不使用其他模型代评分，不关闭 Gate |
| 模型 | GPT-5.4-mini；PPO scorer 必须与执行模型一致 |
| 数据 | RoboSpatial，沿用既定 environment/deployment 划分；此前范围为各 175 题。正式运行前仍需绑定具体 manifest 和内容隔离检查结果，本次未重新生成划分 |
| rollout | 环境阶段每题 4 次、单 pass；沿用部署每题一次的协议。若各 175 题，则为 700 条环境轨迹和 175 条部署轨迹 |
| 执行生成 | 本次最新要求为 medium、每次正常生成 16384 tokens；这是对旧版执行阶段 non-thinking/4096 设置的变更 |
| embedding | `text-embedding-3-large`，现有 v2 YAML 已采用该模型 |
| 工具交互 | 沿用最多 50 步，单次生成截断与总步数耗尽分别处理 |
| 检查目的 | 定位实现和实验设计问题；本次不产生 accuracy，也不能判断提升幅度 |

“单纯把 GPT 名称替换进完整主方法就可以运行”的前提不成立。现有 API baseline 能运行，只能说明生成/检索路径可用，不能证明它具备 PPO 所需的重评分能力。

“记录整个解题过程”可以覆盖全部可观察输入、模型返回、动作、工具结果和知识版本；不能承诺记录 GPT 的完整内部思维链。官方文档说明原始 reasoning tokens 不公开，支持的模型可返回 reasoning summary，后者不能当作原始思维链或 PPO 固定前缀。[OpenAI reasoning 文档](https://developers.openai.com/api/docs/guides/reasoning)

**已确认的阻塞点与适配问题**

| 编号 | 类型 | 事实、证据与影响 |
|---|---|---|
| F1 | 接口能力阻塞 | `OpenAIResponsesProvider` 未实现 `score(request, target_text)`，实际离线调用得到 `ModelCapabilityError: OpenAIResponsesProvider does not implement fixed-target scoring`；GPT-mini 配置也未声明 `FIXED_TARGET_SCORING`。[provider 契约](/home/xiwei.liu/spatialcraft/src/spatialcraft/models/interfaces.py:413)、[模型配置](/home/xiwei.liu/spatialcraft/configs/models/gpt-5.4-mini.yaml:6) |
| F2 | 运行器实现边界 | preflight 要求本地 Qwen；`ExperimentRuntime` 同样在 `config.local is None` 时拒绝启动。离线实例化已复现第二处错误。[preflight](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/run.py:153)、[runtime](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/runtime.py:112) |
| F3 | 与新执行配置冲突 | v2 校验拒绝 `enable_thinking=True`，也拒绝 `execution.accumulation.reasoning_mode=thinking`，两种路径均已离线复现。这是旧 non-thinking 协议的显式约束，不能据此说旧协议实现有 bug。[校验](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/operation_profiles.py:276) |
| F4 | token 配置覆盖风险 | 基于现有 v2 YAML，只把顶层 `max_output_tokens` 改为 16384，解析后的 accumulation/deployment/fallback 仍全部为 4096；因为具体 operation 的配置最后覆盖顶层值。已离线复现。[解析顺序](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/operation_profiles.py:80) |
| F5 | API 移植后的潜在错误 | 主 rollout 每步注入 generation seed；Responses 适配器明确拒绝 seed，已离线复现 payload 报错。当前本地执行路径没有这个问题，但仅替换 provider 会触发它。[seed 注入](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/rollout.py:84)、[API 检查](/home/xiwei.liu/spatialcraft/src/spatialcraft/models/providers/openai_responses.py:203) |
| F6 | API 移植后的潜在错误 | journal binding、tokenizer、Skill learner 及 token 指标多处直接访问 `model.local`。绕过入口检查后仍不能正常使用 API 模型。[绑定](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/runtime_v2.py:124)、[tokenizer](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/runtime_v2.py:194)、[learner](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/learning_v2.py:76) |
| F7 | 配置与日志一致性风险 | 主 Composer 固定写入 `effective_reasoning_mode=instruct` 和 `enable_thinking=False`，却只覆盖生成预算、temperature、top_p，没有应用 API 的阶段 reasoning 映射。若未来只解除入口限制，可能出现模型默认 medium 与日志 instruct 不一致。[Composer](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/runtime_v2.py:229) |
| F8 | 已确认的诊断日志缺口 | `RunJournal` 的失败 attempt 只记录异常类型，不记录异常消息、traceback、失败结束时间；usage 的 provider 失败事件也只保存异常类型。合成异常中携带的详情确实未写入 attempt JSON。[journal](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/journal.py:253)、[usage](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/usage.py:104) |

这些发现不能简单合并为“8 个代码 bug”。F1 是能力缺口；F2/F3 是旧实现的约束；F4 是配置覆盖行为；F5/F6/F7 是适配 GPT 主执行路径时必须处理的问题；F8 是本次调试需求下已经确认的日志不足。8 项离线探针也不等于完成了全项目测试。

**为什么 PPO Gate 不能由普通 GPT 调用替代**

当前协议需要固定历史动作 `a`，在相同历史状态下，仅替换 Skill 上下文，用同一执行模型计算新旧条件下的动作概率比。实现要求保留原动作 token IDs、原始请求、动作前缀及前缀 token IDs，然后以 `raw_model_softmax` 重评分。[动作证据构造](/home/xiwei.liu/spatialcraft/src/spatialcraft/knowledge/skill/learning_v2.py:561)

运行器显式要求 scorer 与 executor 相同，并检查固定目标重评分能力。这项检查是防止方法被意外替换，应保留。[完整方法检查](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/runtime_v2.py:46)

官方 Responses 接口文档描述了模型生成消息的 logprobs，但本次核对未找到能对任意已固定动作、在修改后的 Skill 上下文下进行 teacher-forcing 重评分的公开接口。生成时返回的 logprobs，不能直接提供候选 Skill 条件下同一旧动作的概率。[Responses 接口文档](https://developers.openai.com/api/reference/python/resources/responses/methods/create)

因此，增加 `logprobs` 字段、让模型口头输出置信度、用 LLM judge 打分、换 Qwen 代评分或关闭 Gate，都不能被标记为本次要求的“GPT-5.4-mini＋完整 PPO”。也不能仅在 YAML 中伪声明评分 capability。

若主执行改为 thinking，还需明确历史隐藏推理状态如何参与概率定义。现有 thinking 审计要求可见的固定 `sampled_thinking_prefix`，GPT 的内部 reasoning 无法按该文本协议记录。[thinking prefix 检查](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/rollout.py:138) 这进一步说明：当前问题不是单纯调整 effort 参数，不能通过省略审计就认为协议得到保留。

本报告的边界是“当前项目实现与本次核对的公开接口不能满足要求”；未对用户实际 API 网关进行付费能力探测，也不推断未来接口能力。

**thinking 与 token 配置应如何对齐**

GPT-5.4-mini 官方模型页列出 `medium`，其输出容量允许 16384 上限；本次离线 payload 正向检查也正确得到 `reasoning: {effort: medium}` 和 `max_output_tokens: 16384`。该检查没有发送请求，不能代替实际网关参数验证。[GPT-5.4-mini 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-mini)

16384 限制单次生成，包含 reasoning、可见输出等生成 tokens；不代表能得到 16384 tokens 的可见回答，也不是整条轨迹总预算。达到上限时可能还没有可见动作，应沿用 action recovery 处理。[OpenAI token 预算说明](https://developers.openai.com/api/docs/guides/reasoning)

下表是后续适配时的建议对齐方式，**本次没有修改这些配置**。最新要求优先用于正常执行；辅助阶段沿用此前确认的逐阶段设置。若希望所有辅助调用也一律 medium/16384，应显式记录为另一套配置，不能通过全局值悄悄覆盖。

| 调用阶段 | 当前 v2 设置 | 后续适配目标 / 是否需要改动 |
|---|---|---|
| accumulation / deployment / no-skill fallback | instruct / 4096 | 按本次最新要求改为 medium / 16384；同时处理 F3/F7，而非只改顶层 token 值 |
| rollout summary | thinking / 8192 | GPT thinking 映射为 medium，预算沿用 |
| cross-rollout critique | thinking / 16384 | medium，预算沿用 |
| local experience merge | instruct / 1024 | 显式 none，预算沿用 |
| global experience manage | thinking / 8192 | medium，预算沿用 |
| semantic gradient | thinking / 8192 | medium，预算沿用 |
| gradient aggregation / candidate generation | thinking / 16384 | medium，预算沿用；候选分别计费和记录 |
| task decomposition / experience rewrite | instruct / 1024、2048 | 显式 none，预算沿用 |
| Skill termination / selection judge | instruct / 256 | 显式 none，预算沿用；当前实现包含 LLM applicability judge，选择流程并非完全没有生成调用 |
| truncation recovery / forced final | instruct / 1024、512 | 显式 none，保留小预算及原步骤语义 |
| PPO Gate | 固定目标动作重评分 | 不增加推理生成调用；先解决 F1，且需与新的执行模式保持严格一致 |

知识生成路径已有 provider reasoning 映射，以及 Responses 不传 seed 的处理，可作为未来适配参照；不能据此认为主 rollout 已完成同样适配。[reasoning 映射](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/reasoning_modes_v2.py:9)、[KnowledgeGenerator](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/knowledge_generator.py:54)

另外，不能把 API 的 `seed=42` 当作生成可复现保证。可以固定数据划分、任务顺序和本地随机过程，但本适配器不能向 Responses 传生成 seed。temperature/top_p 也需核对实际发送字段和网关接受情况；日志应区分 requested 与 effective，不能把未发送参数推定为服务端默认值。

**当前过程记录的覆盖范围与缺口**

现有记录框架并非空白。下列路径均是未来有效运行的 `<run>/<dataset>/` 下的现有代码输出约定，**不是本次已经产生的实验结果**：

| 路径 | 可查看内容 |
|---|---|
| `journal.json` | 数据、配置、模型、资源等运行绑定 |
| `stages/tasks/<task>/rollouts/<rollout>/execution/steps/<step>/` | selection、模型请求/返回、recovery、forced final、decision、action、工具交互及状态转换等分步记录 |
| `stages/model_calls/<hash>/inputs.json`、`result.json` | 完整序列化请求和模型返回；成功返回的 provider raw 内容也被保留 |
| `stages/ppo_calls/<hash>/` | 评分请求及评分结果（需要可用 scorer） |
| `stages/knowledge_validation/<hash>/` | 知识生成的 schema 校验、修复信息和 prompt 标识 |
| `rollouts/training/<task>/<rollout>.json`、`rollouts/deployment/<index>.json` | 完整轨迹导出 |
| `experiences/task-<index>.json`、`skills/round-<index>.json` | 随任务演进的 Experience / Skill 版本与更新结果 |
| `checkpoints/` | 每题、演化轮次、待处理演化队列和最终冻结知识状态 |
| `usage/*.json`、`results/usage.json` | 实际调用、缓存复用、token 使用、延迟等审计和汇总 |
| `results/deployment.json`、`results/protocol_metrics.json` | 部署准确率及协议统计；仅在相应阶段执行后才有结果 |

路径依据：[积累与导出](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/accumulation.py:166)、[逐步轨迹](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/rollout.py:72)、[模型审计](/home/xiwei.liu/spatialcraft/src/spatialcraft/experiments/usage.py:96)。工具中间图、深度、掩码等通过 ArtifactStore 保留路径与校验值；排错时必须同时保留实际文件，单独保存 URI 不能替代证据。

为满足此次“能够定位哪一步出错”的要求，后续至少需要补充或核验：

- 失败 attempt 与 usage 保存异常 message、traceback、cause、失败时间；API 错误提取 HTTP 状态码、request ID 和实际可得的限流/超时信息。当前上层 stderr 可能保留 traceback，但结构化日志没有保证。
- 把每次正常生成、恢复、强制作答、知识修复和工具调用串到同一 task/rollout/step；同时核对截断事件与 recovery 不额外占用工具步数、达到 50 步后不再执行新工具。
- 逐次记录实际 reasoning effort、预算、输入上下文规模、reasoning token usage、截断原因；不能只记录 YAML 或固定的 `instruct` 标签。
- 如接口支持且需要可读推理概览，可请求并存储 reasoning summary。当前 payload 只发 effort，没有主动请求 summary；保留 raw response 也无法补出服务端未返回的内容。
- 对 Skill 更新保留父版本、证据轨迹、语义梯度、聚合结果、全部候选、固定目标评分、ratio/advantage/objective、接受/拒绝原因及前后知识快照；应在可运行后用小样本走通一次真实更新，确认路径和内容完整，而非只检查文件名存在。
- 保留工具模型加载/执行耗时及异常时资源信息。GPT 和 embedding 在远端不意味着整个 SpatialCraft 都是 CPU 作业；本地空间工具仍可能需要 GPU。资源选择应由实际工具集合和预检决定。

本次未修改日志实现，也未生成真实 trajectory；旧日志中从未记录的异常细节或内部 reasoning 不能事后恢复。

**哪些问题现在还不能下结论**

当前证据不能证明检测、深度或几何工具必定报错，不能证明 Experience/Skill 必定提高或降低准确率，也不能证明 16384 是最优上限。它只证明上述启动/接口限制和离线可复现行为。模型输出长度、工具成功率、候选拒绝率、知识使用率、上下文膨胀和实际准确率都需要真实轨迹验证。

对提升效果的判断需要对齐模型、reasoning effort、正常生成预算、数据划分、图像输入及评分口径。与旧 non-thinking/4096 direct baseline 直接比较，无法把差异单独归因于 SpatialCraft。优先对照应包含同配置 tool-only ReAct；Experience/Skill 的各自贡献还需要对应消融，单次完整方法运行不足以做因果归因。

正式运行前应复核 environment/deployment 的题目、图像和答案隔离，不能只比较 task ID；绑定当前 `preparation` 的 manifest 和内容隔离报告。此次未重划分数据，也不声称已完成对整个 RoboSpatial 数据集的内容去重验证。

**保留完整要求时的后续顺序**

1. 首先获得同一 GPT 执行模型可用且可验证的固定目标动作重评分能力，并明确 thinking 模式下的动作概率/历史状态语义。这是能力前提，普通代码修复不能创造未提供的接口。
2. 能力满足后，再适配主运行器、scorer、tokenizer/指标、operation reasoning 控制、API seed 处理及有效配置审计，继续保留完整方法的能力检查。
3. 补齐失败日志和知识演化证据链，通过离线契约检查后，用少量真实任务验证完整 4-rollout 学习及至少一次 PPO Gate，再准备正式作业。
4. 正式运行冻结代码与数据 manifest、记录全部有效超参数，完成后按一致口径与 baseline 比较。

**本次交付状态：检查报告完成；8 项离线探针均确认预期行为；无新作业 ID、无新 accuracy、未修改算法/运行代码。**
