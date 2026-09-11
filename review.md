# SpatialCraft 实现修改指南

> 2026-09-11 实施说明：本轮按用户确认推进代码改造，新增 `spatialcraft_v2` 正式执行路径，旧学习协议与配置保留为 `legacy_v1`；严格复现历史工具行为仍须使用当次冻结源码。本文中的 fa72082 链接描述的是审查基线，不代表现有代码仍缺失对应功能。实际入口、方法决策、验证结果与尚未验证的边界见 [v2 实施记录](docs/spatialcraft_v2.md)。文中引用的外部审计/复现文档若只提供 `/Users/mac/...` 路径，不在本工作区，不能当作本轮已核验的证据。

> 基于 2026-09-11 的代码审计，将 23 项发现转化为实施要求、修改位置、配置方案和验收标准。
>
> 审计基线：仓库 Xiwei-web/SpatialCraft，分支 snapshot/20260910-231444，commit **fa72082b124f9827559f310ef42cc099b61f759e**。
>
> 文档性质：**修改指导与验收规范，不表示代码已经完成修改。**

## 1. 目标与适用范围

目标是使实际实验代码实现稿件所定义的双层记忆机制：Experience 提供局部条件—行动指导，Skill 提供可复用的跨步程序；在离线积累阶段学习、整理和演化知识，在部署阶段冻结知识并执行空间推理。

核心实施原则：

1. **在线执行使用 Instruct；需要分析、归纳的离线知识操作按阶段开启 Thinking。**
2. **将方案中的学习算子接入实际实验入口。** 定义类、编写 YAML 或增加 prompt 文件，均不等于功能已接入。
3. **空间 Skill 必须能够调用实际存在的工具完成其程序。**
4. **PPO 评分定义、配置、日志和论文公式必须一致。**
5. **使用真实调用消耗衡量在线成本，分别报告离线学习与在线部署。**

本指南包含三类内容：

| 类型 | 含义 | 实施规则 |
|---|---|---|
| 必须修复 | 方案明确要求、审计确认缺失或接线错误 | 按验收标准完成 |
| 推荐初始配置 | 为落实当前讨论提出的预算、温度或工程策略 | 显式配置并通过环境集验证；不能写成原论文已验证最优值 |
| 方法定义选择 | 稿件与代码之间存在不同但可能合理的算法定义 | 形成版本化决策记录，代码与正文同步；本指南给出推荐方向 |

完整证据见 [代码审计报告](/Users/mac/.codex/.chatgpt-projects/g-p-6a8c06d07d708191bf439d73484c00f6/audit/SpatialCraft-code-audit.md)，组件复现结果见 [reproduction_results.json](/Users/mac/.codex/.chatgpt-projects/g-p-6a8c06d07d708191bf439d73484c00f6/audit/reproduction_results.json)。

## 2. 保留已有正确行为

以下行为应纳入回归验收：

- 同一环境任务执行 4 条独立 rollout；四条共享相同的冻结 Experience/Skill 快照。
- 一组 rollout 全部结束后才更新知识，不能在该组中途改变知识。
- 训练执行温度 0.7、top-p 0.9；部署执行温度 0。
- 执行、检索、重写、Skill 选择和终止判断不能接触 ground truth。
- ground truth 仅用于验证，以及训练任务完成后的离线学习。
- 部署期间不修改知识内容、版本、使用统计或演化队列；运行日志可以正常写入。
- 每个环境步骤只执行一个工具调用；每条轨迹最多 50 次环境/工具交互。
- 截断恢复最多一次，最多 1024 输出 tokens；恢复生成本身不额外消耗环境步骤，恢复后执行工具仍计一步。
- 达到 50 步后允许一次最多 512 tokens、无工具权限的最终回答，并正常验证答案。
- 一个 Skill 最多连续控制 8 个环境步骤。
- Skill 更新版本后，旧版本剩余轨迹不得伪装成新版本行为数据。
- scorer 使用可验证的固定目标 token 概率；不把模型自评或新生成文本冒充 teacher-forced scoring。
- 保留快照、轨迹、工具产物和版本追踪机制。

主要入口：[run.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/run.py) → [runtime.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/runtime.py) → [ProtocolPipeline](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/accumulation.py) / [JournaledRollout](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/rollout.py)。

修改必须作用于这条链路。Fast Qwen 入口也应消费相同的方法配置。

## 3. 修改后的总体流程

### 3.1 环境积累阶段

~~~text
读取环境任务
  ↓
冻结当前 E/K，建立本任务快照
  ↓
多模态任务分解 → embedding 检索 → 多模态经验重写
  ↓
4 条独立 rollout，共享同一快照与同一份重写结果
  ↓
逐轨迹多模态 Summary
  ↓
Cross-Rollout Critique → 0 到 max_ops 个 add/modify
  ↓
在临时 Experience Bank 中依次处理更新
  ├─ add：embedding 相似项检索 → LLM Merge 判断/合并 → 写入
  └─ modify：校验目标 ID/版本 → 更新
  ↓
如果超容量：LLM Global Manage → 校验操作 → 提交
  ↓
Skill 轨迹分段、语义诊断、相关性筛选
  ↓
分别进入 REFINE / DISCOVERY 证据队列
  ↓
满足批次条件后：LLM Aggregate → 3 个候选 → PPO-style Gate
  ↓
接受候选、更新版本、质量淘汰、容量维护、清理旧版本队列
  ↓
提交下一任务使用的知识快照
~~~

Experience 和 Skill 更新可以各自有独立的原子提交边界，但必须记录成功、跳过或失败状态，以及各自依赖的输入快照。

### 3.2 部署阶段

~~~text
加载并冻结最终知识快照
  ↓
多模态任务分解 → embedding 检索 → 多模态经验重写
  ↓
当前无 active Skill：
    从适用集合选择 Top-1，或返回 NONE
  ↓
Instruct executor 生成下一工具调用/最终答案
  ↓
执行工具，将可消费的数值、关系和图像纳入当前观察
  ↓
Instruct 终止判断 / 8 步上限
  ↓
必要时重新选择 Skill；没有适用 Skill 则走 Instruct fallback
  ↓
最终答案、验证结果与完整在线 usage 汇总
~~~

部署不执行 Summary、Critique、Merge、Manage、Semantic Gradient、Aggregate、Candidate Generation、Gate 或知识统计更新。

## 4. 配置、模型角色与 prompt 管理

对应审计：**F15、F18、F23**。

### 4.1 建立按操作区分的配置

修改位置：

- [ExperimentSettings](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/settings.py)
- [ExperimentRuntime](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/runtime.py)
- [LearningBuilders](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/learning.py)
- [RequestBuilder](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/models/request_builder.py)
- [模型角色配置](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/configs/roles/knowledge_builder.yaml)
- [实验配置](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/configs/experiments/qwen35_9b_spatialcraft.yaml)

必须做到：

1. 用操作级配置替代全局 enable_thinking 对所有操作的控制。
2. executor、knowledge builder、scorer、embedding 分别解析角色绑定；允许 executor 和 knowledge builder 使用同一个权重实例，但生成配置独立。
3. 删除“所有输出必须 ≤4096”“辅助温度必须为 0”“embedding 只能 small”等过时硬校验；改为按操作、provider 能力和实验 profile 校验。
4. LearningBuilders.text 接收明确的 operation，按该 operation 解析模式、温度、预算和模板。
5. 实际发出的 request 应与最终解析配置一致，不允许更高层公共预算再次隐式压回 4096。
6. 主实验默认值与消融覆盖分开校验：主实验 N=4 不应阻止显式命名的 rollout-count 消融。
7. 每次运行保存完整 resolved configuration，而不只保存输入 YAML。

推荐接口形式，属于待实现接口：

~~~python
generate_knowledge(
    operation="experience.summary",
    payload=summary_input,
    media=linked_images,
    expected_schema=RolloutSummary,
)
~~~

### 4.2 推荐的阶段配置

下表为当前讨论的实施起点，**尚无预算最优性的实验结论**。Merge/Manage 等新步骤必须先存在真实调用，配置才有意义。

| 操作标识 | 模式 | 单次输出上限 | 温度 | 说明 |
|---|---|---:|---:|---|
| execution.accumulation | Instruct | 4096 | 0.7 | top-p=0.9 |
| execution.deployment | Instruct | 4096 | 0 | 确定性执行配置 |
| experience.summary | Thinking | 8192 | 0.6 | 每条轨迹总结 |
| experience.critique | Thinking | 16384 | 0.6 | 每组任务比较 |
| experience.merge | Instruct | 1024 | 0 | 每次局部合并操作 |
| experience.manage | Thinking | 8192 | 0.6 | 超容量时全库审查 |
| skill.semantic_gradient | Thinking | 8192 | 0.6 | 与 Skill 段关联的诊断 |
| skill.gradient_aggregation | Thinking | 16384 | 0.6 | 每个演化目标一次 |
| skill.candidate_generation | Thinking | 16384 | 0.7 | 每个候选独立预算，3 候选 |
| retrieval.decomposition | Instruct | 1024 | 0 | 每任务一次 |
| retrieval.rewrite | Instruct | 2048 | 0 | 明确是逐条还是批量调用 |
| skill.termination | Instruct | 256 | 0 | 只输出短结构化决定 |
| skill.selection_judge | Instruct | 256 | 0 | 仅选择 LLM 判适用性方案时启用 |
| execution.fallback | Instruct | 4096 | 随阶段 | 训练 0.7，部署 0 |
| execution.recovery | Instruct | 1024 | 0 | 一次额外恢复调用 |
| execution.forced_final | Instruct | 512 | 0 | 禁止工具 |
| skill.ppo_scoring | 不生成 | 不适用 | 不适用 | 固定目标评分，另行定义分布 |

新增操作的 0/0.6 温度是推荐初值，应作为配置显式保存。生成操作的 top-p 也应明确设置，避免自动继承训练 executor 的值。

需要区分：

- **生成总预算**：通常包括 provider 计入输出的 reasoning tokens。
- **最终可见结果**：必须满足 JSON、Experience 长度或 Skill 长度约束。
- **输入上下文预算**：包括图片和历史，不由输出 token 上限控制。

provider 对 Instruct/Thinking 的控制方式需要独立适配和测试。不能只在 prompt 写“关闭思考”就宣称实际关闭了原生 reasoning；不支持的模式应明确报告，不能静默近似后继续沿用同一实验标签。

### 4.3 建议的配置层次

以下是新的配置结构示意，**不是当前快照可以直接读取的配置文件**：

~~~yaml
protocol_version: spatialcraft_v2

roles:
  executor: qwen3.5-9b
  knowledge_builder: qwen3.5-9b
  scorer: qwen3.5-9b
  embedding: text-embedding-3-large

execution:
  rollouts_per_task: 4
  max_environment_steps: 50
  max_skill_horizon: 8
  max_image_pixels: 1048576

experience:
  max_words: 64
  capacity: 100
  merge_cosine_threshold: 0.70
  critique_max_ops: 4
  manage_trigger: capacity_exceeded

retrieval:
  decomposition_min_aspects: 2
  decomposition_max_aspects: 3
  top_k_per_aspect: 3

skill_evolution:
  batch_trajectories: 6
  preferred_low_reward: 3
  preferred_high_reward: 3
  max_parent_skills_per_round: 2
  candidates_per_target: 3
  capacity: 20
  stored_skill_max_tokens: 1024
  ppo_clip_epsilon: 0.2
  acceptance_gain_margin: 0.0

operations:
  experience.summary:
    role: knowledge_builder
    reasoning_mode: thinking
    max_output_tokens: 8192
    temperature: 0.6
  experience.merge:
    role: knowledge_builder
    reasoning_mode: instruct
    max_output_tokens: 1024
    temperature: 0.0
  skill.termination:
    role: knowledge_builder
    reasoning_mode: instruct
    max_output_tokens: 256
    temperature: 0.0
~~~

完整实现应为 4.2 表中全部实际生成操作配置 profile。critique_max_ops=4、stored_skill_max_tokens=1024 是本指南推荐的新增工程限制，需同步到最终实验协议。

### 4.4 Prompt 统一加载

建议增加或接通以下模板职责：

| 类别 | 模板 |
|---|---|
| Experience | summary、critique、merge、manage |
| Retrieval | decomposition、rewrite |
| Skill | semantic_gradient、aggregation、candidate、termination |
| 可选选择器 | applicability judge |

要求：

- 从一个确定的模板来源加载，消除未使用的模板副本。
- 每个模板绑定输入 schema、输出 schema、operation profile 和版本。
- 保存最终渲染 prompt 的 hash；必要时保存可追溯的渲染内容。
- 修改模板后，实际 request 与缓存身份必须变化。
- 保留现有知识优先级：当前视觉/工具证据 > Skill > Experience。
- 结构修复或截断重试属于独立调用，计入预算与日志；不得把失败解析静默当成成功的空更新。
- 知识输出建议至多一次结构修复重试，作为新增可配置策略；其模式和上限单独明确，不与 agent action recovery 混淆。

**验收：** 捕获主 Runtime 发出的真实请求，检查 summary 为 Thinking/8192、merge 为 Instruct/1024、termination 为 Instruct/256、executor 为 Instruct/4096；变更模板或 kb 角色后，请求及缓存身份随之改变。

## 5. Experience 学习与维护

对应审计：**F01–F05、F16**。

### 5.1 多模态轨迹总结

修改：[visual_summarizer.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/visual_summarizer.py)、[learning.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/learning.py)、轨迹输入 schema。

输入至少包含：

- task ID、question、choices、原始图像。
- 逐步 action、工具参数、工具结构化结果和工具图像。
- 每幅图像的来源、step index、artifact ID 与坐标系。
- active Skill 的 ID、版本、定义及 activation segment。
- 本任务检索和注入的 Experience 的 ID、版本、内容。
- 已公开返回的动作解释/推理摘要；不要求补造不存在的 reasoning trace。
- rollout 完成后的 reference answer、verifier outcome 和 reward。

图片应与其说明和所在步骤对应，避免把全部图像无标识地堆到末尾。

建议输出字段：关键观察、关键决策、工具使用、Skill/Experience 的影响、失败原因、证据引用。总结应区分“观察到的事实”和“事后推断”。

**验收：** 构造视觉相同但工具参数不同、或使用不同旧经验的两条轨迹，检查模型输入能够明确区分差异；同时确认 GT 未进入执行请求。

### 5.2 Cross-Rollout Critique 输出多操作

修改：[cross_rollout_critic.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/cross_rollout_critic.py)、[operations.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/operations.py)。

将单一 condition/action 返回值改为结构化操作数组。必须支持零条、多条和 modify：

~~~json
{
  "operations": [
    {
      "type": "add",
      "condition": "When the reference frame is ambiguous",
      "action": "Identify the requested observer before comparing directions.",
      "evidence_refs": ["trajectory-1:step-2"]
    },
    {
      "type": "modify",
      "target_ref": "E17@2",
      "condition": "When comparing depth under a known camera view",
      "action": "Compare consistent depth evidence for both objects.",
      "evidence_refs": ["trajectory-2:step-3"]
    }
  ]
}
~~~

以上为建议 schema，字段名可以适配现有对象，但语义必须保留。

处理要求：

1. 输入四条 summary、任务、验证结果、此前注入的 Experience 内容与版本。
2. 每条经验仅表达局部 condition–action 指导，不扩展成整条 Skill。
3. 两个文本字段合计不超过 64 words。
4. modify 必须引用真实、可更新的版本。
5. 无有效新知识时允许 operations=[]。
6. 禁止将任务答案、实例坐标或对象身份直接记忆为通用经验。
7. 将合法操作应用到临时银行；全部校验通过后再提交。

**验收：** 一次 critique 能新增两条经验并修改一条旧经验；空操作不新增数据；非法 ID 不产生部分写入。

### 5.3 局部 Experience Merge

修改：[consolidator.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/consolidator.py)、[index.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/index.py)、[learning.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/learning.py)。

目标算法：

~~~text
收到 ADD e_new
  ↓
计算 condition + action 的 embedding
  ↓
在当前临时活跃库检索 cosine > 0.70 的已有经验
  ├─ 没有候选：直接加入临时库
  └─ 存在候选：
       将新经验、候选内容和版本交给 LLM
       判断语义重复、互补与条件冲突
       返回合并决定和 ≤64 words 的结果
       校验目标引用
       替代被合并的活跃项，保留来源与历史
  ↓
刷新受影响项的 embedding/index
~~~

边界约定采用“大于 0.70”，与正文措辞统一；阈值等于 0.70 的测试必须固定，不能由不同模块各自解释。

要求：

- **低于容量时也执行局部 Merge。**
- 先检索、再调用 LLM；不能把词重叠当 embedding cosine。
- 相似度只用于寻找候选，不能强迫合并适用条件不同、行动相反的经验。
- LLM 可以判定候选不应合并，保留独立经验；这项细化需在方法描述中明确。
- 合并只消费本次请求提供的合法引用，不能修改模型自行编造的 ID。
- 不允许分别选“最长 condition”和“最长 action”拼接。
- 后续操作必须看见临时库已经发生的变更，避免同一批新增互相重复。
- 内容或 embedding model 改变时，旧向量必须失效。

**验收：**

- 库中只有 2 条经验时，也能触发相似项的 LLM 合并。
- embedding 高相似但字面重叠低的条目可进入 LLM 判断。
- 相似主题、相反条件的两条经验可被保留。
- 没有相似项时不调用 Merge LLM。
- 合并后只有预期活跃项，历史来源完整。

### 5.4 全库 Experience Manage

修改：[maintenance.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/maintenance.py)；建议将 manager 的 LLM 决策职责与统计职责分开。

**触发：** 本轮局部更新处理完后，活跃经验数超过 100。无需为了形式完整而每个任务都做全库 LLM 审查。

输入：完整活跃经验库、稳定 ID/版本、容量、必要来源摘要，以及明确标为描述性信息的使用统计。

输出：merge/delete 操作和简短理由。例如：

~~~json
{
  "operations": [
    {
      "type": "merge",
      "source_refs": ["E12@1", "E23@2"],
      "condition": "When observations use different coordinate frames",
      "action": "Transform both observations into the same frame before comparison.",
      "reason": "Equivalent guidance with complementary frame checks."
    },
    {
      "type": "delete",
      "target_ref": "E45@1",
      "reason": "Duplicates an existing actionable rule."
    }
  ]
}
~~~

质量标准：

- 保留可泛化、可执行、受证据支持的知识。
- 删除明显重复、过度实例化、无行动价值的条目。
- 保留不同条件下的互补规则与空间任务多样性。
- 不将多个 Experience 合并为长篇 procedural Skill。
- archive 可以作为逻辑 delete 的存储实现。

提交规则：

1. 校验引用、版本、重复目标、操作冲突、长度和最终容量。
2. 在工作副本中应用，成功后原子提交。
3. 如果模型未使容量降到上限内，允许有界修正请求。
4. 修正仍失败时，保留先前有效银行并显式记录本次更新失败/跳过；不能无限重试，也不能静默改用原规则排序后标成“LLM 管理成功”。
5. 若另设规则兜底，必须在配置与实验结果中标识其启用次数。

**验收：** 给定 101 条经验，真实调用 manager；完成后活跃数 ≤100、每条 ≤64 words、所有操作可追踪。非法操作或中断不会留下半更新的库。

### 5.5 修正 Experience 统计的含义

不要继续把四条 rollout 共享经验的组内中心化 reward 总和当作经验质量。

至少区分：

| 统计 | 含义 |
|---|---|
| retrieval_count | 被检索出的次数 |
| injection_count | 通过 rewrite 后被放入 executor 上下文的次数 |
| observed_reference_count | 有显式证据表明动作引用了该经验的次数，可选 |
| outcome_association | 注入时的结果关联，不能表述为因果收益 |

不要求 executor 为了统计而每步新增一次长解释调用。缺乏贡献归因时，应把“使用”诚实地命名为“注入”。

全库质量判断优先由 LLM 根据经验内容与证据完成。若后续增加因果收益或对照评估，应单独定义协议与成本，不能沿用会恒等抵消的公式。

**验收：** 使用 reward=[1,0,1,0] 的共享检索案例，不再把总和为零的组内优势声称为每条经验的质量分数；新经验不会仅因未被使用而自动立即删除。

### 5.6 长度约束

在生成、modify、merge、manage 提交之前统一检查：

~~~text
word_count(condition) + word_count(action) ≤ 64
~~~

定义统一 word counter；英语主实验可采用声明过的空白分词规则，避免不同模块各用一种计数方式。失败时请求压缩并重新校验，不直接切断字符串。

## 6. Experience 检索与视觉适配

对应审计：**F06、F07**。

修改：[task_decomposer.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/task_decomposer.py)、[retriever.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/retriever.py)、[contextual_rewriter.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/experience/contextual_rewriter.py)。

### 6.1 Task Decomposition

- 输入 question、choices、任务图像与必要的公开元数据。
- 输出 1–3 个 type/query 对象。简单问题允许单一技术需求，避免为了凑数制造重复查询；2–3 可配置为独立变体。
- Query 描述抽象技术需求，不直接求解任务，也不提前构造完整 Skill。
- 每个任务只分解一次；训练四条 rollout 共享结果。
- 库为空时可以显式短路整个检索分支，记录 skipped_empty_bank。
- JSON 数组长度、字段、非空与重复项均须校验。

建议格式：

~~~json
[
  {
    "type": "reference_frame",
    "query": "Resolve viewpoint-dependent left-right relations."
  },
  {
    "type": "depth_ambiguity",
    "query": "Compare object depth under partial occlusion."
  }
]
~~~

### 6.2 Retrieval

- 统一使用最终选定的 embedding 模型；当前 Table 6 的实施目标为 text-embedding-3-large。
- 每个 aspect 检索 Top-3，按经验 ID/版本合并去重。
- 索引记录模型 ID、维度、文本规范化版本及内容 hash。
- 模型更换后重建索引与缓存，不混用 small/large 的向量空间。
- 经验 cosine 门槛 τ_min 与 Merge 的 0.70 是两个参数，不得互相代用。

τ_min 尚未有可信既定值。允许明确选择：

1. 在环境/验证数据上校准门槛，并将固定值写入实验配置；或
2. 明确采用无门槛 Top-k，并同步删除正文 Eq.(4) 的门槛条件。

禁止保留论文中的有效过滤描述，同时让代码隐式采用 -1.0。

### 6.3 Contextual Rewrite

输入当前视觉任务、检索集合及其 ID/版本；输出每条经验的保留/跳过决定和 condition/action。

~~~json
{
  "items": [
    {
      "source_ref": "E17@2",
      "decision": "keep",
      "condition": "When the requested observer differs from the camera",
      "action": "Identify that observer's axes before comparing left and right."
    },
    {
      "source_ref": "E21@1",
      "decision": "skip",
      "reason": "The task does not require metric distance."
    }
  ]
}
~~~

要求：

- 真实传入图片，不能只提供“visual context”占位文本。
- 保留条件—行动对应关系，不能生成没有条件的泛化建议。
- 不读取 GT，不把最终答案嵌入重写结果。
- rewritten Experience 只存在于本任务上下文，不覆盖冻结银行。
- 可逐条或批量调用；必须固定策略，明确 2048 是单次调用预算，并统计实际调用数。
- 批量处理有利于跨经验去重；若分批，需要最终去重规则与来源保留。

**验收：** 同一问题文本配两种不同视觉情境时，请求包含正确图像；结构不合法的文本不会被直接注入；部署前后银行 hash 不变。

## 7. Skill 语义诊断与聚合

对应审计：**F08、F09、F13**。

### 7.1 以 activation segment 为证据单位

修改：[segments.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/segments.py)、[semantic_gradient.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/semantic_gradient.py)。

每个 segment 记录：

- trajectory ID、task ID、snapshot ID。
- 父 Skill 的 ID/版本。
- 起止步骤、实际 action 与 tool evidence。
- 原始图像和该段相关工具图像。
- 原任务 reward、冻结时确定的 advantage。
- 可用的前后文；不能把不属于该段的动作错误归到当前 Skill。

同一轨迹可有多个 activation segment，但对同一父 Skill 的批次计数最多贡献一个 distinct trajectory。

### 7.2 is_related 由模型判断

建议诊断格式：

~~~json
{
  "skill_ref": "K3@2",
  "trajectory_id": "T8",
  "is_related": false,
  "reason": "The failure arose before this Skill could affect the decision.",
  "activation_update": null,
  "procedure_update": null,
  "termination_update": null,
  "evidence_refs": ["T8:step-4"]
}
~~~

- 保留 LLM 的真实布尔值，禁止使用 bool(skill_ref) 替代。
- false 可以不附更新建议，schema 应允许。
- 被执行不代表对结果有影响；成功和失败都可判定无关。
- is_related=false 不自动等于“需要新 Skill”。
- 独立表达“相关但无需修改”，避免迫使模型为正确 Skill 编造改进。

### 7.3 区分待诊断与可演化队列

如果 Table 6 的 B=6 指“相关轨迹”，必须在计数中体现语义相关性：

1. 原始轨迹先成为候选证据。
2. 诊断结果按 trajectory/segment/Skill version 缓存。
3. 只有符合相关性定义的 distinct trajectories 进入 eligible batch。
4. 同一父 Skill 多次激活、同一诊断重试或任务恢复不能重复计数。
5. 无关证据保留审计记录，但不强行凑满批次。

### 7.4 实现 3 low + 3 high

当前 reward 为二元值，low=0、high=1。

- 同一父 Skill 的同一版本下，优先选 3 条失败、3 条成功相关轨迹。
- 某侧不足但总数 ≥6 时，从另一侧补齐，记录不足情况。
- 同一 reward 组内可使用 FIFO，保证确定性。
- 总数不足 6 时保留队列，不把重复 segment 当新轨迹补足。
- batch 选取不能重新计算原任务 baseline。
- 父 Skill 版本失效后，旧版本剩余 eligible/pending 项归档，不继承给新版本。

### 7.5 独立 LLM Gradient Aggregation

修改：[gradient_aggregator.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/gradient_aggregator.py)。

输入同一父 Skill/版本的诊断集合、证据支持、reward/advantage；输出：

- activation 修改方向。
- procedure 修改方向。
- termination 修改方向。
- 支持各方向的证据引用。
- 冲突意见的处置与不确定性。
- 没有足够支持时的 no_change 决定。

聚合器必须进行真实 LLM 调用；字符串去重可以保留为预处理。

冲突不能一概投票消除：若不同情境分别支持相反行动，应保留对应条件，形成条件分支；证据不足的冲突应明确暂不修改。

**验收：** false 不被覆盖；已有三成功时批次满足 3/3；“总是先估深度”和“从不先估深度”不会未经分析原样进入一条无条件 procedure；三个候选共享同一份有效聚合结果。

## 8. Skill REFINE、DISCOVERY 与候选生成

对应审计：**F10、F16**。

修改：[evolution_queue.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/evolution_queue.py)、[candidate_generator.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/candidate_generator.py)、[learning.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/learning.py)。

### 8.1 REFINE 路径

- 从一个确定父 Skill 与其聚合梯度产生 3 个独立候选。
- 三个候选均以同一个原父 Skill 为基准，不能候选 2 继承候选 1。
- 候选分别保留生成种子、来源、版本关系和输出。
- 有效候选通过 Gate 后最多接受其中一个。
- 接受后同一 skill_id 的版本加一，旧版本标记 superseded。
- 新版本统计按声明的规则重新开始；旧版本的统计仍保留用于审计。

### 8.2 DISCOVERY 路径

增加能够真正到达 parent=None 的路径，不能只保留未调用的 NEW 分支。

推荐的首个可验证实现：

1. 收集历史中 active_skill=None 的实际 fallback 段。
2. 根据重复出现的空间技术需求组织 discovery 证据桶。
3. LLM 判断该需求是否已被现有 Skill 覆盖、是否需要新的程序。
4. 达到声明的证据量后生成新 Skill，采用独立的新 skill_id。
5. 候选与当时真实 NONE 行为上下文比较，通过 Gate 后 pool.add。

重要约束：

- “某个已激活 Skill 对结果无关”不能自动将当时轨迹视作 NONE 行为。
- 如果历史生成时有父 Skill，删除它以后得到的是新的反事实上下文，不能假称为原 behavior denominator。
- 可以从更广泛轨迹提议新 Skill，但严格的 NONE Gate 需要匹配的历史证据；不足时等待符合既有数据协议的后续采样。
- 新 Skill 的激活条件应足够明确，不能全部写成通用“遇到空间问题时”。
- 不把修改名称或替换旧版本当成 discovery 成功。

推荐新增工程限制：每轮最多处理 2 个演化目标，其中最多 1 个 discovery；父 Skill 上限仍不超过 2。该调度限制是新建议，应显式配置并在论文中说明，不能隐藏额外生成预算。

### 8.3 候选结构与长度

每个候选必须包含 activation/initiation、procedure/policy、termination、来源证据和演化类型。

推荐最终长度约束：

~~~text
tokens(skill.format_for_prompt(), declared_tokenizer) ≤ 1024
~~~

- 16384 是每次候选生成允许使用的总输出预算。
- 1024 是最终会被注入 executor 的 Skill 文本长度。
- 以指定 tokenizer 计数，并记录 tokenizer ID。
- 如果部署换 tokenizer，重新统计实际注入长度；不静默改写冻结 Skill。
- 不能存储模型的完整 Thinking 过程作为 Skill。
- 长度失败时压缩后重新校验；不得破坏 procedure 的必要条件和终止要求。

**验收：** 除六个种子外出现可追溯的新活跃 skill_id；REFINE 保留父子关系；所有候选评分共用原父 Skill；Skill 文本长度通过实际 tokenizer 校验。

## 9. Skill 选择、终止与质量淘汰

对应审计：**F11、F12、F14**。

### 9.1 允许 NONE 的选择器

修改：[selector.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/selector.py)、[skill_controller.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/agent/skill_controller.py)。

采用“适用性筛选 → Top-1”：

~~~text
applicable = {k: applicability(k, current_state)}
selected = argmax similarity(current_state, activation(k))
           over applicable
若 applicable 为空，则 selected = NONE
~~~

适用性可以采用校准后的 embedding 阈值，或预算受限的 LLM 判断。正式实验只选定并声明一种主策略。

要求：

- minimum_score 在 embedding 分支中也必须生效。
- Skill 阈值与 Experience Merge 的 0.70 无关，应独立校准。
- 使用环境/验证数据选择阈值，不能用部署标签调参。
- 如果 LLM 只审查检索 shortlist，需声明候选范围；不能称为全池适用性审查。
- 缺少校准值时可继续开发其他模块，但正式 profile 不能隐式使用“永远 argmax”。
- 非空池仍必须能返回 NONE。
- 有 active Skill 时保持其控制，直到终止或达到 horizon。

### 9.2 终止判断输入

修改：[termination_controller.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/agent/termination_controller.py)、[lifecycle.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/lifecycle.py)、[learning.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/learning.py)。

输入安全 task view、原始图像、当前 Skill、当前动作、最新有效观察，以及与终止条件直接相关的历史证据。

输出短 JSON：

~~~json
{
  "terminate": true,
  "reason_code": "condition_satisfied"
}
~~~

- 使用 Instruct/256/T=0。
- 最终答案、达到 8 步等确定条件可以直接由代码结束 Skill，无需额外判断调用。
- 终止 Skill 不等于终止整个任务。
- 任务图像和问题不能因为 after_step 丢弃 task 而消失。
- 条件是否满足不能通过隐藏 GT 判断。
- 是否允许同一 Skill 再次激活应明确记录；单纯再次激活不自动视为错误。

### 9.3 质量淘汰与容量维护分开

修改：[maintenance.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/maintenance.py)、[statistics.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/statistics.py)。

在每轮学习更新后依次：

1. 更新有明确归属的 Skill 使用和 gain 统计。
2. 按成熟度保护规则淘汰负贡献 Skill；零优势不等于无效，不作为质量删除条件。
3. 去除功能和适用条件都重复的 Skill。
4. 若仍超过 20，再按明确分数排序归档。
5. 清理失效版本队列并提交。

要求：

- 活跃数 ≤20 也执行正常质量淘汰。
- 不因六个初始种子分数为 0 而立即删除全部种子。
- 成熟度默认至少 3 条不同、具有非零组内优势的轨迹；同组全成功或全失败时优势都为零，这类轨迹不能单凭零值触发删除。不要把一条轨迹里三次激活当三条独立证据。
- 未成熟项目的保护与超容量时的处理顺序必须明确。
- 语义去重应比较 activation/procedure/termination 的功能，避免仅用词重叠删除不同条件的程序。
- 允许采用 embedding 候选检索加语义判断；新增调用须有独立 profile 与日志。
- pool 为空时保持 NONE fallback 和 discovery 可用。

**验收：** 成熟、负 gain 的 Skill 在容量未满时被归档；初始种子在没有证据时仍可用；相似措辞但不同条件的 Skill 不被误删；旧版本缓冲不继承。

## 10. PPO-style Gate：先固定定义，再修改实现

对应审计：**F17**。

修改：[ppo_gate.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/ppo_gate.py)、[credit_assignment.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/knowledge/skill/credit_assignment.py)、[target_logprob.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/models/scoring/target_logprob.py)、[transformers_local.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/models/providers/transformers_local.py)、[learning.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/learning.py)。

### 10.1 必须形成方法决策记录

至少记录以下选项，并保存到每次运行：

| 决策 | 本指南推荐方向 | 必须同步的描述 |
|---|---|---|
| 接受条件 | 保留相对提升 ΔJ > margin，margin=0 | 正文绝对 J>0 改为正 gain |
| baseline | v2 使用原任务全部 rollout 的 reward 均值（主实验 N=4） | 明写 group baseline，不再称 running baseline；重组3/3后不重算 |
| action 分数聚合 | v2 使用生成动作 token logprob 求和 | 历史 mean 变体保留在 legacy 路径，不混为同一方法 |
| 评分分布 | v2 使用底层模型原始 softmax likelihood surrogate | 不称为 temperature/top-p 采样行为分布的严格 importance ratio |
| action target | 保存并使用实际历史 action token span | 规范化重序列化如需保留，单列评分变体 |
| 轨迹/segment 权重 | v2 先对每条轨迹中归属该目标的动作取均值，再对轨迹等权平均 | 不再额外除以轨迹长度；segment只是归属证据单位 |

这些推荐并不同时等价于原稿所有公式。选定后应更新正文、附录、实现与测试，再命名新实验版本。

### 10.2 固定历史 action

保存并复用：

- 原始 multimodal request 与图片/工具产物版本。
- 原始 action tokens、对应 token IDs 与边界。
- 生成时的 executor、tokenizer、chat template 和解码设置。
- 实际 behavior Skill ID/版本或 NONE。
- 若采用 Thinking executor 变体，保存 provider 实际返回的完整 reasoning prefix。

对于同一个样本，只替换指定的 Skill prompt slot。图片、历史、工具 schema、action target 和固定 prefix 均保持一致。

Instruct 下如有 action 前的可见解释，也必须明确如何划分和处理；不能一边声称只评 action，一边把任意前导解释混入目标。

### 10.3 区分两种 ratio

完整序列 likelihood ratio：

~~~text
log_ratio = Σ_j log p_new(a_j | context, a_<j)
          − Σ_j log p_old(a_j | context, a_<j)
ratio = exp(log_ratio)
~~~

当前长度归一化 ratio：

~~~text
log_ratio = mean_j log p_new(a_j | ...)
          − mean_j log p_old(a_j | ...)
ratio = exp(log_ratio)
~~~

必须做到：

- 两者使用不同配置标识和实验名称。
- 不用 schema 中的 mean_logprob 字段冒充 sequence logprob。
- 在 log 空间进行必要数值处理；任何额外截断需记录，不能悄悄改变 PPO penalty。
- 保留逐 token trace，确保可以独立重算 Gate。
- candidate 和 parent 的 target tokenization 一致；不一致则拒绝该评分。

### 10.4 评分分布与 sampling 设置

若评估底层模型 likelihood，明确说明它是 PPO-style likelihood surrogate；生成数据使用 T=0.7/top-p=0.9，不意味着原始 softmax 分数就是实际采样分布概率。

若要求真实行为分布 ratio，必须同时处理温度变换、top-p 的支持集与归一化；候选下历史 token 概率为零时也需要明确规则。

不要通过更换 scorer 模型获得另一个模型的概率后，继续声称它是 executor 的行为概率。没有所需 scoring 能力的 backbone 应在运行前明确判为不支持该 Gate，或使用单独标记的替代实验。

### 10.5 Advantage 与接受条件

若使用 group baseline：

~~~text
baseline(task) = mean(reward of the original four rollouts)
advantage(trajectory) = reward(trajectory) − baseline(task)
~~~

- 在原四条 rollout 完成后确定并持久化。
- FIFO/3+3 选批、跨任务组批、恢复运行时不得重算。
- 是否除 trajectory length、是否按 activation segment 平均，必须有明确的统一公式。
- 候选排名与父 Skill 的目标使用同一组样本和同一套权重。
- 推荐接受条件：best_candidate_J − parent_J > 0。
- 全部 advantage 为 0 时，记录无有效相对评分信号；不强行接受候选。

### 10.6 PPO 验收案例

| 案例 | 预期 |
|---|---|
| parent 与 candidate 相同 | ratio=1，gain=0，不接受 |
| 100 tokens、每 token logprob 提升 0.1 | sequence 模式 ratio≈exp(10)；mean 模式≈exp(0.1) |
| parent J=-1，candidate J=-0.9 | gain 模式可接受，并明确区别于绝对 J>0 |
| 高/低 advantage 样本 | clip 对两种符号均符合固定公式 |
| candidate 改变 tokenizer | 不能比较为相同 action |
| NONE discovery | 分母使用历史真实 NONE request |
| 全零 advantage | 明确记录无信号，不接受 |
| Thinking prefix 存在 | 同一 prefix 固定，仅对声明的 action span 评分 |
| 轨迹长度不同 | 手算权重结果与实现一致 |

## 11. 空间工具必须形成可执行的三维链路

对应审计：**F19、F20**。

目标链路：

~~~text
图像 → reconstruction（带有效掩码、内参、外参及尺度状态）
     → detection / segmentation
     → 按明确的 source-to-processed 映射对齐 mask 与 point map
     → masked world points / visible-surface median centroid
     → object frame / geometric measurement
     → 可消费的数值、关系与可视化
~~~

### 11.1 统一坐标、单位与 artifact 契约

每个空间结果应明确：

- source image/frame ID、输入尺寸、处理尺寸及两者的像素中心映射。
- 坐标系 ID、轴方向、左右手约定；独立重建调用不能复用同一 world ID 冒充已经对齐。
- camera-to-world 或 world-to-camera 的方向；相机采用 OpenCV 右手坐标 x右/y下/z前。
- 长度/角度单位、尺度来源与验证状态。model-estimated metric 不等于已用真实尺寸校准。
- 有效像素/点掩码、置信度的原始语义、失败状态；非概率的置信分数不能夹进[0,1]后宣称是概率。
- artifact 内容类型、字段、来源及可用的下游操作；NPZ 自身保存必要坐标和尺度元数据。

**方法界限：** 当前真实 checkpoint 为 DA3-BASE；本地上游 README 区分 any-view 与 metric/nested 系列，因此不能继续无条件标注 meter。使用 `reconstruction_unit`/`unverified`，经 MoGe 对齐后标记为 `meter`/`estimated_metric`。MoGe 只提供另一个模型的尺度估计，其一致性不是绝对尺度准确性的证明。

像素中心、resize/crop 与 normalized intrinsics 的转换必须分别说明。MoGe normalized intrinsics 转到整数像素中心需要 `[[W,0,-0.5],[0,H,-0.5],[0,0,1]] @ K`。DA3 当前入口仅接受尺寸相同的真实多视角图像并使用明确的 resize 模式，避免无法审计的混合尺寸 batch center-crop；未来若扩展此输入范围，应先保留实际变换，不能根据最终尺寸猜测裁剪位置。

“世界坐标”不自动代表重力对齐。没有可靠的重力、地面或其他定向证据时，不能强制声明 +Y向上、地面Y=0、重力BEV。可见点中位数是可见表面的稳健位置估计，不是完整物体质心；其 OBB 也不代表被遮挡部分的完整体积。

### 11.2 工具修改清单

| 模块 | 必须具备的能力 | 关键验收与边界 |
|---|---|---|
| Geometry | 2D/3D距离、3D夹角、向量旋转、SE(3)变换、投影/反投影，消费持久化点集 | 解析点距离和变换精确一致；反投影深度为camera-z；相机后方点、零向量、非SE(3)输入明确拒绝/不可用 |
| Reconstruct | world point map、绝对frame ID、有效掩码、内参、外参方向、source-to-processed映射、自包含NPZ | 合成场景坐标满足解析值；真实场景按预先声明容差验证，不要求模型重建完全精确 |
| SAM3 / Mask | source图像尺寸核对、masked world points和中位数3D位置；mask面积、median、IoU及边界统计 | mask不因resize/crop错位；空mask/无效深度返回available=false；2D精确bbox与percentile bbox字段分别定义 |
| Pose | 结合orientation、reconstruction、mask构造object_to_world/front/centroid/可见表面OBB及源图投影 | 使用有来源的角度约定；低置信度、退化轴或不足点返回不可用；PCA/OBB本身不能决定语义front |
| Scale | 对齐同一source image的有效深度，报告correction_factor、overlap与disagreement | 已知尺度偏差可恢复；高分歧不强制校正；若输出校正副本，points/depth/camera translations一起缩放，不修改原artifact |
| Graph | 同时提供scene relation graph和附录数值序列plot模式 | 关系边进入structured_output，长图可按artifact分页查询；plot输出有效样本统计和可见PNG，缺失样本显示断点 |
| Draw / OCR / Motion | 公开schema与实际输出对应；线/框/点绘制、OCR区域及注释、光流区域mask/有效性/median | 源图不变；flow保持像素位移语义，不冒充米制相机运动 |

Pose 当前采用具名 `orient_anything_v1_722_projection_v1` 约定，对应本地722输出checkpoint及上游 `get_proj2D_XYZ`。以解析案例验证投影及正交性，只证明约定一致；真实物体朝向仍是 `model_estimated`。+Z为语义front估计，+Y为语义bottom，+X=+Y×+Z；外部给定轴必须另行标注来源。

Scale 当前按一个明确frame估计 `median(depth_m/depth_recon)`，分歧为 `median(abs(depth_m/(factor*depth_recon)-1))`。单帧比例应用到同一重建gauge需明确记录，不能写成多帧共识。默认最少32有效像素、最大分歧0.25是可配置工程初值，不是已验证最优参数。

全部附录API的实际状态见 [tool_api_matrix.md](docs/tool_api_matrix.md)。尚未公开的点提示/视频分割、语义地面识别、重力BEV等不能写成已实现；纯软件缺项也不得由近似名称代替。当前0–1 normalized conversion与草稿0–1000定义不同，必须按最终接口同步论文。

### 11.3 Artifact 必须能被使用

修改范围包括 StateBuilder、ContextComposer、工具注册及各工具消费者。

1. 小型JSON、数值和关系边直接放入structured_output。
2. 重建大数组保存为自包含NPZ；mask/pose/scale通过`reconstruction_uri`、绝对`frame_index`消费。masked point集通过`points_uri`进入geometry。
3. 大型关系图保存完整JSON，提供分页与entity/relation过滤；不能只返回edge_count。
4. 图像预览与实际数值结果同时保留，说明source image、像素空间及其对应关系。Graph数值plot和Pose overlay必须作为IMAGE artifact进入模型可见媒体。
5. 禁止无界展开点云到prompt；点集预览及关系分页有明确上限与truncated标记。

**端到端验收：** 合成场景验证三维中心、坐标变换、尺度校正、图关系与artifact接线；同时使用真实工具后端进行最小组合验证，检查模型实际收到对应数值/图像。真实感知允许声明容差和失败，不以纯mock证明真实几何效果，也不要求每一道空间题必须走重建链路。

## 12. 在线成本与实验指标

对应审计：**F21**。

修改：[efficiency.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/evaluation/efficiency.py)、[journal.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/journal.py)、[accumulation.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/accumulation.py)、provider usage 解析。

### 12.1 按操作记录真实调用

每次模型调用至少记录：

| 字段 | 用途 |
|---|---|
| run/task/trajectory ID | 归属 |
| phase、operation、role | 区分离线与在线职责 |
| model、tokenizer、prompt/config version | 复现 |
| requested/effective reasoning mode | 模式核验 |
| max_output_tokens、finish_reason | 截断与预算分析 |
| input_tokens、output_tokens | 真实 usage |
| reasoning_tokens、cached_input_tokens | provider 支持时记录 |
| latency、retry index、cache hit | 实际执行成本 |
| status、parse/validation outcome | 区分成功、跳过与失败 |

注意：

- 未提供的 token 字段记 unavailable/null，不能记成 0。
- reasoning tokens 若已包含在 output_tokens 中，不再重复相加。
- 缓存回放的历史 usage 与本次新增实际调用成本分开。
- 本地模型记录 GPU 时间等资源指标；没有实际计费信息时不伪造美元成本。

### 12.2 在线成本范围

~~~text
online_cost(task)
  = decomposition
  + rewrite（全部实际调用）
  + selection judge（如果启用）
  + executor normal calls
  + termination calls
  + recovery / forced-final calls
  + online retries
  + embeddings
  + spatial tools
~~~

token 指标只对产生/消耗模型 tokens 的调用汇总；工具开销单独以时间/资源记录，再按声明的方法构成综合成本。

四条训练 rollout 共享的一次 decomposition/rewrite 不应重复记四次；摊销到 trajectory 时明确分配规则。

### 12.3 报告指标

至少报告：

- Accuracy 与按任务类别的 Accuracy。
- 每任务真实 input/output/reasoning tokens。
- 每任务模型调用数、工具调用数与总耗时。
- 截断率、恢复率、结构解析失败率、工具参数错误率。
- 重复工具调用率，注明“重复”的判定方式。
- 离线建库总成本。
- 部署每任务成本。
- 给定部署任务数 M 时的摊销成本：
  offline_cost/M + mean_online_cost。
- Skill/Experience 存储规模、实际注入长度与覆盖率。

若报告节省，需要与同 backbone、同任务、同工具和同预算协议的基线比较。基础设施故障不能伪装成模型 reward=0，也不能静默从统计分母中移除。

**验收：** 将一个任务的所有调用手工求和，与自动报表一致；包含 rewrite、termination、重试和缓存案例；state word count 不再标成真实 total tokens。

2026-09-11 实施补充：`results/usage.json` 已给出 per-task、独立调用数、观测在线均值及 `offline/M + mean_online`，缺失用量保留 null。`results/protocol_metrics.json` 单独汇总动作解析/恢复/截断/重复工具调用及知识注入统计。注入长度来自已提交 transition 保存的请求，使用执行器 tokenizer，覆盖不到无动作失败调用或额外 recovery；不能当作全部计费 input tokens。active 库大小按 canonical JSON 字节定义，不含索引/归档/文件系统。实际 GPU 资源计时与美元账单尚未纳入自动报告，调用 latency 求和不是任务墙钟。


## 13. Baseline、消融与跨模型实验

对应审计：**F18、F22**。

修改：[baselines.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/evaluation/baselines.py)、[ablations.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/evaluation/ablations.py)、[runtime.py](https://github.com/Xiwei-web/SpatialCraft/blob/fa72082b124f9827559f310ef42cc099b61f759e/src/spatialcraft/experiments/runtime.py)。

### 13.1 Baseline 必须有真实执行器

每个论文报告的 baseline 至少交付：

- 实际入口或 executor。
- 对应算法的核心组件与学习规则。
- 数据划分、backbone、工具权限、预算与重复运行设置。
- 与原方法不同的适配说明。
- 完整运行日志和结果路径。

不能将 SpatialCraft 简单关掉几个开关就默认为忠实复现所有外部方法。XSKILL baseline 需要实现其声明的 Experience/Skill 两层机制，或明确命名为部分机制变体。

### 13.2 消融必须改变实际行为

| 消融 | 应观察到的变化 |
|---|---|
| no_experience | 无经验学习/检索/重写/注入，Skill 分支按声明保留 |
| no_skill | 无 Skill 选择/终止/演化/注入，executor 正常回退 |
| no_decomposition | 不调用分解模型，使用明确的原问题查询 |
| no_rewrite | 不调用重写模型，直接使用原经验内容 |
| no_visual_summary | 保留可比的文本总结流程，仅移除总结中的视觉输入 |
| no_critique | 必须明确定义替代提取策略，不能与 no_experience 混为一谈 |
| no_local_merge | 新增前不执行局部语义合并，全库管理仍按配置运行 |
| no_manager | 无 LLM 全库管理；容量兜底策略明确记录 |
| no_semantic_gradient | 明确定义候选生成的替代证据来源 |
| no_gradient_aggregation | 候选看到原诊断集合，不执行独立聚合调用 |
| no_ppo_gate | 保留候选生成，采用事先声明的候选接纳规则 |
| no_score_pruning | 保留分数记录，关闭分数淘汰；容量处理单独定义 |
| static_seed_skills | 只有固定六个种子，无演化 |
| mode/budget ablation | 只改变声明的阶段模式或预算，保持其他机制一致 |

对于移除某个产生下游输入的模块，必须定义替代路径，不能让整条分支意外失效却仍声称是单因素消融。

### 13.3 跨模型与能力限制

- kb 和 executor 角色可独立配置。
- scorer 必须与所声称的评分策略对应。
- 不支持固定目标评分的模型，不能借用另一个模型概率后继续标为同一个严格 Gate。
- 跨模型部署使用冻结知识，记录源 kb 模型、目标 executor、tokenizer 与模式。
- 不因为 API direct/ReAct baseline 已可运行，就宣称完整 SpatialCraft 的所有 backbone 已接入。

**验收：** 从主 Runtime 运行最小任务，比较各消融真实调用清单、知识更新和注入内容；不能只检查 overrides 字典。

## 14. 测试、恢复与历史结果管理

### 14.1 三层验收

| 层级 | 内容 | 完成标准 |
|---|---|---|
| 组件测试 | schema、合并、队列、选择、Gate、长度和统计 | 可控输入得到指定行为 |
| 主流程集成 | 真实 Runtime 配置与 request 构造，模型结果可用测试替身控制 | 证明模块已接线，模式/预算/媒体/调用次数正确 |
| 小规模真实实验 | 实际模型、embedding、图片和工具后端 | 验证上下文、预算、工具组合和输出质量可运行 |

三层结果分开报告。语法检查或 mock 测试通过，不代表真实空间后端或模型效果已验证。

### 14.2 必须加入的回归案例

- [ ] 未超容量也触发提交前 LLM Merge。
- [ ] 超容量触发 LLM Manage，非法操作不产生部分提交。
- [ ] 一次 critique 支持多个 add 与有效 modify，也允许零操作。
- [ ] 72-word Experience 被拒绝或压缩。
- [ ] decomposition/rewrite/termination 实际收到对应图像。
- [ ] is_related=false 不被覆盖，且不计为相关演化证据。
- [ ] 具备两侧样本时按 3/3 选批，distinct trajectory 不重复计数。
- [ ] 冲突梯度由聚合模型处理，条件差异得以保留。
- [ ] NONE 轨迹可到达 discovery，产生新的 skill_id。
- [ ] 相似度为 −1 的 Skill 不因非空池而被强制激活。
- [ ] 容量未满时成熟负贡献 Skill 仍会淘汰，初始种子得到保护。
- [ ] 生成总预算与最终存储长度分别校验。
- [ ] 概率比、advantage、轨迹权重与 gain 可独立重算。
- [ ] graph edges 对模型可见，大型 artifact 可被下游工具计算。
- [ ] 三维几何与对象坐标系通过解析场景及真实后端检查。
- [ ] 四条 rollout 使用同一冻结快照，部署前后知识 hash 一致。
- [ ] 恢复、forced-final、GT 隔离和旧版本队列规则不退化。
- [ ] 实际总 usage 与按调用手工统计一致。
- [ ] 每个消融改变声明的行为，而非只改变配置名称。

### 14.3 更新旧测试

现有测试中有对 FIFO、全局 4096、固定 small 和全辅助温度 0 的断言。

这些测试表达旧协议。修改时应：

1. 保留对应历史 profile 的测试或历史记录。
2. 为新 profile 编写新的行为验收。
3. 替换不再适用的固定假设，不能通过删掉检查掩盖问题。
4. 在有依赖的环境运行完整相关测试，保存命令、环境和结果。
5. 只在新失败或未解决风险需要时扩大测试范围。

### 14.4 缓存、恢复与版本

新缓存身份至少包含 operation、完整有效配置、模板、输入快照/内容、模型与评分模式。

- 预算、模式、模板或数学定义变更后，不复用不兼容的旧结果。
- 中途恢复不重复调用已提交操作，也不重复累加使用统计。
- 新的 bank/skill 操作使用可检查的提交记录，避免恢复造成重复 merge/add。
- 历史结果保留原 source/config/protocol 身份。
- 新方法默认从新的学习运行开始；若迁移旧知识作为初始化，必须单独命名并披露。
- 不修改旧结果的配置元数据，把它重新标记为新方法结果。

## 15. 实施顺序与阶段交付物

| 阶段 | 工作 | 交付物与进入下一阶段条件 |
|---|---|---|
| A：统一协议和接口 | 模型角色、操作配置、schema、prompt loader、PPO 决策记录 | resolved config 可打印；所有操作有明确契约 |
| B：Experience 闭环 | summary、critique、local merge、global manage、长度、统计、视觉检索重写 | 新增、修改、合并、删除均从主入口可达 |
| C：Skill 闭环 | 相关性、3/3 队列、LLM aggregate、REFINE/NEW、NONE、pruning、termination | 新 ID 可产生，旧 Skill 可正常淘汰 |
| D：评分对齐 | action span、ratio、baseline、权重、gain、scorer 能力检查 | Gate 数值可重算、方法定义与代码一致 |
| E：空间工具闭环 | point map/mask/3D geometry/pose/scale、artifact 消费 | 至少一个依赖真实三维计算的任务端到端通过 |
| F：实验与成本 | usage 汇总、消融、baseline、跨模型入口、恢复验证 | 各对照实际可运行，完整成本可审计 |
| G：小规模验证后正式运行 | 固定 profile、检查截断率与知识质量、冻结版本 | 论文表格只填写对应版本的实际结果 |

阶段 E 的接口和纯几何实现可以与 B/C 独立推进；正式性能实验应在相关必需阶段完成后进行。该顺序是实施依赖说明，不要求创建额外任务或代理。

## 16. 论文、配置与实现同步清单

- [ ] Embedding 统一为最终选择；当前主目标为 large，并更新附录中的 small。
- [ ] Summary/Critique 温度统一，不再同时出现 0 与 0.6。
- [ ] 删除“全部调用统一 4096”的过时描述，加入分阶段表。
- [ ] 明确生成预算包含哪些 token，最终 Experience/Skill 长度单独列示。
- [ ] 明确局部 Merge 和全库 Manage 均由 LLM 执行各自的语义职责。
- [ ] 明确 3/3 选批、样本不足时补齐规则、相关性定义。
- [ ] 明确 discovery 证据、NONE 行为对照和每轮预算。
- [ ] 明确 NONE 适用性判断及校准方式。
- [ ] 明确负贡献淘汰的成熟度门槛，并保护零优势的中性 Skill。
- [ ] 统一 ratio、baseline、权重与“严格正 gain”公式。
- [ ] 将条件 action scoring 与完整行为概率的关系写清。
- [ ] 附录每个工具 API 均有实际能力对应；未实现能力不写成已用。
- [ ] 调整“不依赖外部空间工具”等与当前任务设定冲突的表述。
- [ ] 区分知识只读与运行日志写入。
- [ ] 将完整在线成本、离线成本和摊销方法写入实验部分。
- [ ] 明确基线适配和消融替代策略，标注未完成的模型/方法。
- [ ] 记录每张结果表对应的 source commit、protocol 和配置版本。

## 17. 审计问题到修改任务的完整映射

| 审计 ID | 问题 | 本指南位置 |
|---|---|---|
| F01 | 局部 Merge 缺失 embedding+LLM | 5.3 |
| F02 | 全库 Manage 缺少 LLM | 5.4 |
| F03 | Experience 效用相互抵消 | 5.5 |
| F04 | Critique 仅单条 ADD、MODIFY 未接入 | 5.2 |
| F05 | Summary 轨迹信息不完整 | 5.1 |
| F06 | 分解无图像、数量及门槛未对齐 | 6.1–6.2 |
| F07 | Rewrite 无图像/结构保证 | 6.3 |
| F08 | is_related 被硬编码 | 7.1–7.3 |
| F09 | Aggregation 无 LLM | 7.5 |
| F10 | DISCOVERY 不可到达 | 8.2 |
| F11 | Score Pruning 未接入 | 9.3 |
| F12 | 强制 Top-1，无 NONE | 9.1 |
| F13 | FIFO 未实现 3/3 | 7.4 |
| F14 | Termination 缺少任务上下文 | 9.2 |
| F15 | 全局模式与 token 校验阻碍分阶段配置 | 4.1–4.3 |
| F16 | Experience/Skill 长度缺少约束 | 5.6、8.3 |
| F17 | PPO 公式和实现定义不一致 | 10 |
| F18 | 模型角色、embedding、多 backbone 未接通 | 4.1、6.2、13.3 |
| F19 | 空间工具能力少于附录 | 11.1–11.2 |
| F20 | Graph/artifact 结果不可消费 | 11.3 |
| F21 | 在线成本统计不完整 | 12 |
| F22 | Baseline/ablation 仅有配置清单 | 13 |
| F23 | Prompt 文件未进入实际请求 | 4.4 |

## 18. 最终完成标准

修改完成必须同时满足：

1. 全部必需学习操作从实际实验入口可达，并有操作日志证明调用发生。
2. 新增、修改、合并、删除、发现 Skill、淘汰 Skill 均有可复现案例。
3. 原本正确的冻结、GT 隔离、恢复、版本和工具步数规则全部保持。
4. 分阶段模式、预算、最终长度和模型角色已在实际 request 中验证。
5. PPO 数学定义与实现一致，能够从保存的 trace 独立重算。
6. 空间工具支持论文所声明的关键组合，输出能被执行模型和后续工具消费。
7. 论文报告的 baseline 与 ablation 均有实际运行路径。
8. 完整在线成本与离线学习成本可以按调用重算。
9. 组件测试、主流程集成测试、小规模真实验证分别给出结果与限制。
10. 论文、配置、代码和实验结果绑定到同一个明确的方法版本。

只有满足这些标准，才能将新版本作为完整 SpatialCraft 方法用于正式比较。
