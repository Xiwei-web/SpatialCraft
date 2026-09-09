# SpatialCraft 已确认协议与运行入口

当前工作区的trajectory预算策略以 `docs/action_recovery_v2.md` 为准：token截断使用action recovery，工具步数耗尽才forced-final。本文中旧输出超限收尾描述和已提交快照记录属于历史协议。

更新：2026-09-07，当前协议为 **instruct / thinking=false / 单次输出 4096 / Skill generation 4096**，替代此前 1024/32768-token thinking 联调配置。用户希望知识压缩为可直接执行的 procedural Skills，减少在线重复推理。其余 6 条相关 trajectory / 最多 2 parents、50/8 steps、FIFO 调度等不变。Embedding 已连通，六个空间工具已通过独立真实调用；**尚未运行正式 benchmark**。
主配置：`configs/experiments/qwen35_9b_spatialcraft.yaml`。入口：`python -m spatialcraft.experiments.run`。
不要将旧 `pipelines/accumulation.py` 的通用示例默认 builder 当作本协议入口。

## 1. 协议落实

| 要求 | 实际行为 | 主要代码 |
|---|---|---|
| SMA 思路的约 50/50 划分 | 复用 seed42 已准备的 task-instance 分层划分；不重划数据 | `experiments/protocol.py`, `run.py` |
| 每题 4 independent rollouts，不做 10 pass | 1 pass；每题冻结 E/K，4 个不同 seed，各自独立 state 和工具历史 | `experiments/accumulation.py`, `rollout.py` |
| Experience 每 subtask top3 | task 开始分解、检索、rewrite/filter 一次；去重后合并所有适用项，4 条共享；不在每 step 重检索 | `knowledge/experience/{retriever,contextual_rewriter}.py` |
| 同一 embedding 模型 | Experience 与 Skill 共用 `text-embedding-3-small` 的归一化向量空间及磁盘缓存 | `models/providers/openai_embeddings.py`, `knowledge/experience/index.py` |
| 单一动态 Skill | 开始/终止后对 state 与 initiation 计算余弦 argmax；期间持续激活；终止后允许再次选中同一 Skill | `knowledge/skill/{selector,lifecycle}.py`, `agent/skill_controller.py` |
| 两分支共用轨迹 | 每题 4 条都做视觉 summary 和 cross-rollout critique；相同 trajectory IDs 同时进入当前 batch Skill 更新 | `experiments/learning.py` |
| per-parent batch evolution | 每题 4 rollouts 屏障后检查一次：每 parent 至少 6 条不同相关轨迹，FIFO 取 6 条；每轮最多演化 2 个 parents，尾批不足 6 不强行演化 | `experiments/{accumulation,evolution_queue,learning}.py` |
| segment attribution | 只诊断该 Skill 的连续激活窗口；分别归因 initiation/procedure/termination；同 Skill 再激活算新窗口 | `knowledge/skill/{segments,semantic_gradient,credit_assignment}.py` |
| 每次 3 个候选与 PPO Gate | 同一旧 Skill、同一 aggregate 生成 3 个真实 LLM 候选，逐一严格评分；仅最佳且超过正收益 margin 的候选生效 | `knowledge/skill/{candidate_generator,ppo_gate}.py` |
| Binary reward | verifier 的 correctness 转为 0/1；基础设施故障停止重试，不混入错误答案；步数/输出预算耗尽为 truncated、reward=0 | `rollout/reward.py`, `experiments/rollout.py` |
| 冲突优先级 | 提示词明确 Visual/Tool Evidence > Active Skill > Experience | `agent/context_composer.py` |
| 容量 E≤100 / K≤20 | 限制 active 项；超限执行合并/归档或 Online Score+去重裁剪；保留历史版本和维护记录 | 两个 `knowledge/*/maintenance.py` |
| 冻结部署 | 只能使用已提交的最终 snapshot；拒绝 training task IDs、其他 benchmark；每题单次评估，不更新 E/K/统计 | `experiments/accumulation.py` |

模型角色在此入口明确绑定：executor、summary/critique/rewrite、semantic-gradient/candidate/termination builder、PPO scorer 均使用本地 Qwen3.5-9B；只有 embedding 使用 OpenAI。不使用全局角色配置中的其他付费生成模型作为 fallback。
六个 seed Skills 从 `src/spatialcraft/resources/skills/seed_skills.yaml` 加载，每个 benchmark 独立初始化 E/K，不跨 benchmark 串用知识。

严格 teacher-forced PPO 使用历史 action 的固定目标 token，在保留相同图像、历史、工具 schema 和 instruct 模板的前提下只替换当前 Skill 提示槽。
工具 action 使用 Qwen 原生调用格式，不包含调用前的解释；答案 action 保留解析后的原始答案文本，由 verifier 提取 `Final Answer`。原始输出仍留作审计。
当前评分为 `P(action | state, Skill)`，`ppo_thinking_mode=action_only`，不要求或注入 sampled thinking 前缀。旧 fixed-prefix thinking 评分实现与测试保留，但不用于这次 instruct 实验。Qwen 模板的静态空 thinking 分隔符不等于生成了 thinking。
比率为 `exp(mean_logprob_new - mean_logprob_old)`，应用 PPO clipping；候选 objective 必须 **> old objective + margin**，等于阈值也拒绝。
task baseline 是该题 4 次 reward 均值；per-step advantage 为 `(reward-baseline)/trajectory_length`，Skill 的 Online Score 使用按激活次数累计的 average_gain。
上述 evidence 优先级是模型提示约束，不是程序对视觉事实正确性的保证。

## 2. 新确认参数与调度细节

| 参数 | 当前默认 |
|---|---|
| evolution_batch_trajectories | B=6 条不同 trajectory / parent；不是 6 道题、6 个 segment 或全局攒 6 条 |
| max_parent_skills_per_round | 2；每个入选 parent 仍生成 3 个候选，单轮最多 6 个候选 |
| max_steps | 整条 trajectory 上限 50 个环境交互 step |
| skill_max_lifetime | 一个 Skill 最多连续控制 8 steps，之后重新 selection，不结束整条 trajectory |
| max_output_tokens / skill_generation_max_tokens | 均为 4096；每个候选独立的一次生成也为 4096 |
| accumulation sampling | temperature=0.7，top_p=0.9，不同 seed |
| thinking | 关闭；所有 executor/auxiliary/candidate 调用显式 enable_thinking=false，使用 instruct 模式 |

每道题的 4 条轨迹全部完成后才更新 E/K 和调度队列；同一条 trajectory 即使多次激活某个 Skill，在其队列内只计 1 条。它可以分别进入多个实际调用过的 Skill 队列，诊断和 PPO 仍仅使用各自 active segments。
每轮优先选择最早有待处理轨迹的 eligible parents，平局按 parent reference 排序；每个选中 parent 消耗最早 6 条。未通过 PPO 的 batch 也视为已评估，不重复用同批轨迹持续重试；未选中的 parent 和不足 6 条的尾批保持 pending。
队列按 `skill_id@version` 隔离：若 parent 成功更新或被 prune，旧版本剩余条目转为审计历史，不当作新版本实际控制过的轨迹。若未更新，剩余条目继续等待新轨迹凑足 6。训练结束不额外绕过每轮 2-parent 上限，也不强制演化不足 6 条的尾批；最终 pending 队列会保存。
每条 queue entry 保留原四-rollout task 的 advantage、trajectory ID/哈希及不可变 commit 路径。即便六条跨多个 task，也不能按这六条重新估算 baseline。候选/gradient/PPO audit 可追溯到所消费的精确六条轨迹。

step 目前定义为一次 agent action 与对应 observation 的 transition；同一 action 的多个 tool calls 合并计 1 step，final action 也计一次。decomposition、rewrite、summary、critique、termination judge、candidate 和 PPO scoring 不占环境 step。
输出命中 4096 上限时不自动加预算：executor 标为 truncated、reward=0；knowledge builder 不提交不完整内容，报错保留断点供检查。旧 thinking 联调日志保留不覆盖；新的 instruct 协议使用独立 run 目录，不能跳过协议绑定复用旧轨迹。

补充确认已落实：训练 top_p=0.9；deployment temperature=0；PPO epsilon=0.2、positive margin=0（仍要求严格正收益）；辅助 summary/critique/termination temperature=0，以显式 `auxiliary_temperature` 配置执行，候选生成仍使用训练 temperature=0.7；Skill 更新后旧版本未消费轨迹只存档，不归入新版本队列。decomposition/rewrite/semantic-gradient 继续共用 temperature=0 的辅助构建入口。
当前仅单图像素预算尚待确认：暂保留 65536–1048576，`image_max_pixels=1048576` 不视为用户已同意。候选拒绝后的队列消费和上述 step 计数定义已显式记录，若要改变需新建协议版本。

另外，现有 decomposer 最多保留 6 个 subtask，因此最多 18 个去重前 Experience 候选；rewrite 可以过滤不适用项。
Experience 超限先仅合并 condition/action 完全一致项，再按 utility/使用次数归档；未引入未经验证的语义合并。
Skill 超限先按 average_gain、frequency 排序，再按 initiation/policy/termination 的 token Jaccard≥0.9 去冗余，最后裁至 20。
这些维护细节属于当前明确实现，后续如改用语义去重，需要新建 run 版本。

## 3. 数据与指标

| benchmark | environment | training rollouts | deployment |
|---|---:|---:|---:|
| RoboSpatial | 175 | 700 | 175 |
| ERQA | 200 | 800 | 200 |
| Omni3D | 251 | 1004 | 250 |
| 合计 | 626 | 2504 | 625 |

沿用 `/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1` 的固定数据。deployment 报 overall accuracy 和 question_type/原始 answer_type 分类 accuracy，不以 deployment 分数挑选知识快照。
这是实例级划分，不声称与 SMA 未公开 split IDs 完全相同，也不是 image-disjoint；已记录的图像/题目重复见 `qwen35_three_benchmarks.md`。

Binary reward 不等于对所有答案都做字符串逐字相等：MC/yes-no 先规范化，numeric 沿用项目 abs=1e-3、rel=1e-2 默认容差，RoboSpatial pointing 沿用已有 mask-hit 判定。尚未宣称这些细节与 SMA 官方 evaluator 完全一致；正式比较前须确认。
executor、retrieval/rewrite、termination 不接收 GT；training 结束后的 summary/critique 可以使用 training reference/reward 作监督。deployment GT 仅交给 verifier。模型仍可能在训练经验中记忆题目，prompt 抽象要求不能替代去重评估。

## 4. OpenAI embedding 接口

专用 provider 懒加载：构造配置/运行离线检查均不要求 key，不访问网络；真正 cache miss 时才读取环境变量 `OPENAI_API_KEY`。
模型配置在 `configs/models/text-embedding-3-small.yaml`，通用角色在 `configs/roles/embedding.yaml`，不回退到 Qwen/hash embedding。
根据 OpenAI 官方接口采用默认 1536 维，显式 `encoding_format=float`，验证返回的 model/index/维度/有限性并 L2 归一化。
每条文本最多 8192 tokens；请求按最多 64 条且总量不超过 300000 tokens 分批。超长文本报错，不静默截断改变检索语义。
缓存按 endpoint+model+维度+归一化版本+文本 hash 隔离；存向量/hash，不保存 key 或原始 embedding 文本。Skill/Experience 共享缓存，索引拒绝不同空间混用。
每次返回成功的 API 请求另记 `embedding_cache/usage/*.json` 的 token 用量；收费以 OpenAI 账单为准，崩溃重试或 SDK 内部重试不能保证 exactly-once 计费。
接口依据：[Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)、[Create embeddings](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)。本轮使用 openai-docs skill 核对接口，未测试账户权限/余额/真实返回。

## 5. 日志与恢复

```text
/l/users/xiwei.liu/spatialcraftLog/
├── preparation/qwen35_9b_seed42_v1/     # 固定输入、分层划分、private GT、原始 manifest
├── smoke/                            # 先前真实 provider 联通性记录，非正式成绩
├── validation/protocol-20260907/      # 本轮 preflight、GPU provider 验证、测试结果
└── runs/qwen35_9b_protocol_v2/        # 新协议正式执行时才创建（当前未开始）
    ├── embedding_cache/              # 共享向量缓存 + usage/ 请求用量
    ├── robospatial/                  # erqa/、omni3d/ 同样结构、独立知识
    │   ├── journal.json              # 代码/资源/模型/协议/数据绑定
    │   ├── stages/                   # inputs、attempts、原子 result（恢复权威记录）
    │   │   ├── tasks/                # retrieve、4 rollouts 的逐步工具/模型/状态、Experience 更新
    │   │   ├── evolution/            # 每个 task 屏障的队列/聚合/候选/NP-PPO/维护提交
    │   │   ├── model_calls/          # 原始 request/response，含图像引用及 token usage
    │   │   ├── ppo_calls/            # 固定 target + 完整 token IDs/logprobs
    │   │   └── deployment/           # 只读评估调用
    │   ├── rollouts/{training,deployment}/
    │   ├── experiences/              # 每 task 的 bank、summary、critique、维护 audit
    │   ├── skills/                   # round-*.json；各 parent 的六轨迹/gradients/候选/PPO/队列 audit
    │   ├── tool_store/               # 工具原始结果、图像/数组 artifacts 和 SHA-256
    │   ├── checkpoints/             # task/round 快照 + pending_evolution.json + frozen_deployment.json
    │   └── results/deployment.json   # 总体及分类 accuracy
    ├── erqa/
    └── omni3d/
```

同一命令、同一输出路径重复运行会验证并复用已提交的阶段；未完成的模型/工具阶段重试。task/pipeline 锁防止两个进程推进同一 run。
运行前绑定当前代码、seed/prompt/config、库版本、tokenizer/processor、工具源码，以及准备数据的 manifest；`--execute` 首次 API 前还校验图像及模型/工具权重 SHA-256。
修改配置、代码、数据、依赖、权重时拒绝恢复旧目录，应新建版本；不通过篡改 journal 强行续跑。
外部调用完成但结果未落盘时可能重复执行；不是 exactly-once。正式工具只处理本地图像/分析 artifacts，不授权外部写操作。
原准备 manifest 的代码版本保持不动，新运行器只复用已固定的数据，再绑定当前代码；不要为了新版运行器去覆盖旧 preparation。

## 6. 命令与验收范围

不需要 key 的检查：

```bash
conda activate spatialcraft
cd /home/xiwei.liu/spatialcraft
export PYTHONPATH=/home/xiwei.liu/spatialcraft/src
python -m pytest -q
python -m spatialcraft.experiments.run \
  --report /l/users/xiwei.liu/spatialcraftLog/validation/protocol-v2-confirmed-20260907/preflight.json
```

后续用户在终端安全设置 key、确认暂定参数，并在 Slurm GPU allocation 中完成小规模真实工具/embedding 验收后，正式启动及恢复使用同一命令：

```bash
python -m spatialcraft.experiments.run --execute \
  --output /l/users/xiwei.liu/spatialcraftLog/runs/qwen35_9b_protocol_v2
```

这个命令会依次完成 RoboSpatial、ERQA、Omni3D 的 accumulation → 冻结 deployment，**会付费调用 embedding**；当前未执行。不要把 key 放进 YAML、代码、聊天或日志。

v2 验收新增了每 parent 六轨迹门槛、重复激活去重、两 parent 上限、FIFO/尾批、原四轨迹 baseline、fixed-thinking action mask、1024 总预算和 50/8-step 恢复测试。补充确认后的最新结果为 72 项通过，见 `validation/protocol-v2-confirmed-20260907/unit-tests.xml`；旧 v1/v2 报告保留不覆盖。
实际 A100 40GB 上的 Qwen9 已在 **thinking=true** 下通过 seed 重放、带图固定 thinking 条件动作评分和原生 XML 工具调用；问答/工具调用分别用 114/98 个输出 tokens，均在 1024 内闭合。脚本 `scripts/inference/check_protocol_provider.py --thinking`，日志见新版 validation。使用合成图，不是完整空间工具链。
尚待设置 key 后验证 OpenAI 连接、真实空间模型组合的显存/接口、长多图上下文、LLM schema 输出稳定性及小规模故障恢复；在这些通过前不能称“全链路实验已验收”或报告正式 accuracy。
