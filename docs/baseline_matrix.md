# 可执行基线与复现范围

本文件区分真实执行路径、方法适配和完成的验证。新增记忆基线共享 SpatialCraft 的真实模型、空间工具、动作恢复和验证器，但使用独立的 `MemoryBaselinePipeline` 管理学习、检索与冻结部署；不通过一组开关将完整 SpatialCraft 冒充其他论文方法。

所有新增记忆基线均保存 `faithful_reproduction=false`、`fidelity=paper_inspired_spatial_adapter`。它们是具备实际机制的空间任务适配版本，不代表已复现原论文全部设置、调参和成绩。当前已有 CPU 测试；尚未运行这些新增路径的 GPU/API benchmark，不能填写正式性能结果。

| 方法标识 | 实际学习与检索路径 | 与原方案的边界 |
|---|---|---|
| `rag_demonstrations` | 环境集执行时不检索记忆；仅把验证成功的公开任务、动作参数、工具观察及最终回答加入演示库。部署时按原问题 embedding Top-k 检索，无反思/重写。 | 在同一空间工具协议下构建演示库；不等于任何特定 RAG 论文复现。长示例超出明确保存预算时拒绝保存并记录。 |
| `memp_reflection` | 对每条完成的成功/失败轨迹独立调用 LLM，生成零或一条条件＋程序记忆；直接语义检索。 | 不做跨 rollout Critique、Skill 语义梯度或 PPO；使用统一空间执行器的 MemP-inspired 适配。 |
| `memrl_reward` | 使用独立反思记忆；先语义候选检索，再按 `(1-w)cosine + w empirical_return` 重排。历史注入后的结果更新经验平均值。 | 描述性经验结果关联，不宣称因果收益，也不是原论文 Q-learning 公式的忠实实现。 |
| `sma_procedure` | 仅成功轨迹产生程序记忆；其后训练注入结果校准可靠度；按 `max(0,cosine) × (reward_sum+1)/(count+2)` 排序。 | 明确采用本适配的 Laplace 平滑；单 pass、最终快照，不采用原论文多轮最佳 checkpoint 协议。 |
| `xskill_dual_memory` | 视觉 Experience 的 Summary/Critique/Merge/Manage，另有独立 LLM workflow add/modify；双分支均实际更新与部署。 | 不执行语义梯度/PPO。工作流适配为 SpatialCraft initiation/policy/termination 字段并放入共享执行槽；不宣称原 XSkill 文档形态完全一致。 |
| `skill_pro_sequence` | 空 Experience 库、六种子 Skill；调用真实已接线的语义诊断、候选演化和固定动作 sequence gate；只有 Skill 学习。 | 固定动作 raw-model likelihood surrogate 和空间工具适配；要求实际 sequence scorer/evolver，不能用空回调声称复现。 |
| `memrl_gt` | **Unavailable，启动明确拒绝。** | 当前没有确定 GT 变体的算法与 oracle 协议。不得默认把部署答案传入检索后混入普通准确率表。 |

已有 CoT/direct 与 Tools-only ReAct 的执行入口仍分别是 `spatialcraft.experiments.run_api_baseline` 和 `spatialcraft.experiments.run_react_baseline`。这些已有 API 路径不等于记忆基线或完整方法已经在所有 backbone 上完成运行。

## 运行方式

先在项目根目录查看可用分支，不加载 GPU 或访问 API：

```bash
PYTHONPATH=src python -m spatialcraft.experiments.run_memory_baseline --list-methods
```

指定已确认的 `spatialcraft_v2` 实验配置、准备目录和新的运行输出目录；不传 `--execute` 时只做离线 preflight：

```bash
PYTHONPATH=src python -m spatialcraft.experiments.run_memory_baseline \
  --method xskill_dual_memory \
  --config /absolute/path/to/v2-config.yaml \
  --preparation /absolute/path/to/preparation \
  --output /absolute/path/to/new-baseline-run \
  --datasets robospatial
```

`--execute` 启用真实模型、工具和付费 embedding，遵循已有 Slurm/GPU 与凭据入口。`--pilot-tasks N` 只运行训练诊断子集，禁止附带部署评估。`--memory-capacity`、`--skill-capacity`、`--top-k` 改变声明的基线参数并写入绑定，不能在同一日志目录无记录切换。

## 共用协议与审计

- 训练默认每题四条共享冻结知识的独立轨迹；全部完成后才学习，部署每题一条轨迹。具体次数、采样、图像预算、工具预算、backbone、kb/scorer 和 tokenizer 由 resolved 配置绑定。
- 执行和检索使用去掉标准答案的任务视图。离线反思可以读取已完成环境任务的答案与验证；部署仅运行冻结检索/执行。
- 独立 runner 禁止混入完整方法消融。若要研究适配本身的参数，必须修改明确的基线配置并使用新运行目录。
- MemP/MemRL/SMA 的记忆以及 XSkill workflow 超容量时采用明确的“最早创建项优先归档”策略，历史项保留。该规则是基线适配配置，不是完整方法的 LLM Experience Manage 替代。
- 冻结部署不修改记忆内容、版本或统计。检索过程可正常写运行日志；同一阶段从 journal 恢复不重做已提交调用。
- Skill-Pro 分支要求 `ratio_mode=sequence` 和真实 `evolve_skills` 回调。完整 scorer/候选接受的数值行为由共享 Skill 演化测试覆盖，基线测试额外检查路径实际调用且 Experience 始终为空。

关键产物相对于每个 benchmark 运行目录：

| 路径 | 内容 |
|---|---|
| `journal.json` | 数据、模型、代码、运行配置和基线适配身份 |
| `stages/memory_baseline/descriptor/` | 具体方法、可用性、复现边界与参数 |
| `stages/memory_baseline/training/<task>/` | 检索、四轨迹、记忆/工作流/Skill 更新及原因 |
| `stages/memory_baseline/frozen/` | 最终只读知识快照及未消费演化状态 |
| `stages/memory_baseline/deployment/<task>/` | 部署检索及完整执行轨迹 |
| `results/memory_baseline.json` | 带适配标签的 count/correct/accuracy 与快照 ID |
| `usage/`、`results/usage.json` | 实际 generation/scoring/embedding/tool 调用，含在线/离线、缓存复用与未知用量覆盖 |

CPU 验证入口：

```bash
PYTHONPATH=src python -m pytest -q tests/test_memory_baselines_v2.py
```

Skill-Pro adapter 还为已提交的训练轨迹写入 `stages/tasks/<task>/rollouts/<index>/complete/` 同内容别名，供共享演化队列恢复；该别名不执行额外推理。CLI 支持 RoboSpatial、ERQA、Omni3D、SAT、ViewSpatial，并可读取已有三数据集准备目录。

这些测试验证六条方法路径的更新差异、共享执行回调、效用重排、四轨迹屏障、断点复用、部署冻结、训练/部署 ID 隔离和不支持变体的明确拒绝。它们不替代真实 provider 与全 benchmark 实验。


## Backbone 支持范围

上述六条记忆基线的 `run_memory_baseline` 入口目前使用本地 v2 Runtime，支持 Qwen3.5-9B 与 Qwen3.6-27B；不支持 API-only executor。配置远程 knowledge builder 不等于支持远程 executor。独立 GPT direct、tool-only ReAct 或 `RAG/` 的 API baseline 使用各自入口，不能据此声称这六种记忆适配已经支持 API backbone。该边界属于当前实现范围，尚未执行全 baseline 实验。

## 二次复审后的示例预算与容量

`rag_demonstrations` 保存专用语义表示，排除 raw response、token IDs、评分输入及运行时间戳；原始 journal 仍保留这些复现字段。正式 Runtime 用实际执行器 tokenizer 限制完整注入文本（默认 1024 tokens）。无 tokenizer 的通用测试适配明确回退为 256 words，不把 word 当 token。超长示例仍会拒绝，`results/memory_construction.json` 与 journal 的 construction 阶段报告成功轨迹、候选示例、拒绝/保存/当前有效示例数及部署命中；这不保证所有成功轨迹都适合当前预算。

Skill-Pro 的 `--skill-capacity` 在创建 Runtime 前写入唯一有效 settings，并检查已绑定 evolver 容量一致。六种子协议拒绝小于 6 的容量；容量 10 的回归实际把超容 Skill 池修剪到 10。详情见 [复审说明](review_acdcaec_fixes.md)。
