# Trajectory 与生成预算：action_recovery_v2

本策略已按用户2026-09-09的新要求实现于工作区。正式实验入口JournaledRollout和通用ExecutionLoop共用 `src/spatialcraft/agent/decision.py`；已提交作业执行其冻结快照，工作区修改不会自动改写旧作业。

## 控制规则

- 正常生成仍由实验配置限制为每次4096 tokens。每次生成一个下一步工具调用，或证据充分时给出最终答案。Prompt要求简洁，禁止重复推理和一次输出整条trajectory；关闭并行工具声明，动作检查限定每步一个工具调用。
- 一次generation达到长度上限，先检查完整action。支持完整原生Qwen工具块、完整JSON工具调用及已闭合参数；工具名称、数量和参数须通过校验。截断后的完整工具块可以直接执行，不执行后面的残缺草稿。
- 对被截断的文本最终答案，要求显式 `Final Answer: ...` 完整行（有换行边界），或闭合的 `{"final_answer": "..."}` JSON对象。无法确认完整性的最后一行、未闭合thinking、半截工具参数中的“Final Answer”都不当作最终答案。
- 无完整合法action时，同一step最多发起一次action-only recovery，预算1024 tokens（若原call上限更小则取较小值），temperature=0、thinking=false。保留原视觉输入、历史observations、Experience、Skill和工具schema；不把被截断的推理草稿追加给模型。Recovery只需一个合法工具动作或显式 `Final Answer: ...`；普通推理文字不能作为恢复成功的最终答案。
- Recovery不增加environment step。恢复出的工具动作执行并形成observation之后，才进入下一步。Recovery达到自己的长度上限时也可使用已完整形成的action；若仍无合法action，则FAILED、reward=0并结束该rollout，无第二次recovery。
- 最多执行50次工具交互。达到上限后，基于最新state的全部视觉证据、工具结果、对话历史及记忆，额外执行一次512-token forced-final-answer call，禁用工具、temperature=0、thinking=false。给出答案后进行验证并结束，答案错误也正常结束；若输出为空、仍残缺或请求工具，则记录失败并结束，不追加recovery。

轨迹允许“50个工具transition + 1个terminal answer transition”。终止答案记录不消耗environment step；`metadata.environment_steps` 和 `total_tool_calls` 最大均为50。`Transition.step_index`仍是连续的存储序号，以保持既有数据结构和PPO归因。

## 记录与恢复

每个step的journal保存：

- `.../steps/NNNN/model/`：普通调用。
- `.../steps/NNNN/recovery/model/`：当前step唯一的恢复调用（若需要）。
- `.../steps/0050/forced_final/model/`：50步耗尽后的唯一最终作答调用。
- `.../steps/NNNN/decision/`：已提交的解析/恢复决定；重启读取该决定，不重复已完成的调用。
- `.../steps/NNNN/tools/000/`：实际工具结果；相同journal重放不重复执行工具。

每条trajectory的 `metadata.generation_events` 包括kind（normal/recovery/forced_final）、state/step、生成上限、实际输入/输出token数、finish_reason和token_truncated标记。`metadata.call_counts` 分别统计normal_llm_calls、recovery_calls、tool_calls、token_truncations和forced_final_answers。这些是executor决策调用的逻辑统计；Skill终止、Experience/Skill构建的辅助模型调用仍在model_calls及其请求/usage中单独审计。

每个transition记录实际产生action的 `model_request` 与call_kind。PPO使用实际请求；thinking常规动作仍固定已完成的reasoning prefix，禁用thinking的recovery/forced-final动作按action-only评分。

只有明确分类的模型动作失败可作为训练零分继续后续rollout；CUDA、网络、工具运行与存储错误仍传播，不能混成零分数据。失败generation的原始输出和成功恢复结果均保留。辅助Experience/Skill结构化输出仍使用独立的knowledge compact recovery，不会触发trajectory强制作答。

## 新旧运行边界

`runtime.dataset()`将policy、max_tool_steps、actions_per_step、recovery次数及两个补充token预算写入 `journal.json` 的 `binding.trajectory_control`。新策略不可静默续接旧策略数据；应建立新运行目录及冻结快照，不能用普通代码修复声明将两种算法协议混合。历史日志、知识快照和已提交作业保持原样。

## 验证

- 完整回归 **153项通过（32.35秒）**，Ruff通过。
- 行为测试覆盖完整/残缺截断动作、参数合法性、恢复后继续工具交互、显式最终答案、最多一次恢复、恢复再次截断、50次工具交互后收尾、错误最终答案、禁止额外工具、失败分类、断点重放及统计。
- 原有50步/8步Skill寿命测试现在验证50个工具动作和一个终止答案，且最终Skill无需额外模型调用即可终止。
- 真实Omni3D配置离线初始化通过，含journal创建/提交/重开；代码测试使用确定性模型与mock工具，不等同于新策略已完成真实GPU benchmark。
- 验证目录：`/l/users/xiwei.liu/spatialcraftLog/validation/action_recovery_v2_20260909/`。
