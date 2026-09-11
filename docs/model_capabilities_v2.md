# SpatialCraft v2 模型能力与配置契约

此说明对应 `spatialcraft_v2` 实现。模式适配与 payload 测试不等同于真实服务端验证；API 请求拒绝会作为实际失败记录，不会静默换模式或模型。

## 角色与预算

- executor 必须匹配 `backbone`。当前正式 runtime 使用本地 executor，供历史 action 的固定 token 评分。
- scorer 必须与 executor 使用同一 alias、同一模型实例与权重；不同 scorer 不是该 Gate 的行为模型。
- knowledge_builder 可复用 executor，也可绑定另一个配置。另一个本地模型须声明安全加载布局；API 模型须声明下面的模式映射。
- `operations.<operation>` 控制真实输出预算、温度、top-p、模式和有界 JSON 修正。修正使用同角色的 Instruct 配置，修正预算也接受模型能力检查。
- `max_input_tokens` 由本地 tokenizer/processor 按实际 token 输入执行；当前 API/vLLM 不能宣称执行相同的精确输入上限，配置该字段会在运行前拒绝。
- `execution.fallback` 用于实际 NONE 状态。它可以覆盖输出预算；温度与 top-p 继承当前 accumulation/deployment 配置。两阶段最终值保存为 `effective_fallback_profiles`。

`skill_generation_max_tokens` 是 `skill.candidate_generation` 的默认生成预算，操作级显式值优先；最终配置保存实际生效值。最终存储 Skill 的 tokenizer 长度由 `skill_options.stored_skill_max_tokens` 单独限制。

## 显式模式控制

本地 Transformers 路径使用实际 chat-template 参数 `enable_thinking`。API/vLLM 必须在模型 YAML 的 `metadata.v2_reasoning_modes` 中同时声明两个模式。请求审计保存 `reasoning_control` 和 `reasoning_control_validation`。

Responses / OpenAI-compatible Chat 的示例：

```yaml
metadata:
  v2_reasoning_modes:
    instruct:
      reasoning_effort: none
    thinking:
      reasoning_effort: medium
```

Responses adapter 发送 `reasoning.effort`；Chat adapter 发送 `reasoning_effort`。这里声明的是该具体 endpoint 的配置契约，不能根据模型名称推断其支持哪些值。仓库中的 GPT-5.4/mini 通用配置带有上述映射，其状态是 `configured_unverified_endpoint`。Responses 不支持 seed 参数：保留 `requested_generation_seed`，记录 `seed_applied=false`，实际请求不发送 seed。独立候选仍发送独立请求，但不能声称 API seed 已控制生成。

Gemini Native 可显式映射预算：

```yaml
metadata:
  v2_reasoning_modes:
    instruct:
      thinking_config:
        thinking_budget: 0
    thinking:
      thinking_config:
        thinking_budget: 2048
```

映射经现有 SDK 的 `GenerateContentConfig.thinking_config` 发送；两者都不继承模型原来的 `reasoning_effort`。当前实现要求 Instruct 对应 `thinking_budget=0`，不会把 MINIMAL thinking 标成关闭思考。若具体模型不支持关闭思考，它不能用于要求 Instruct 的操作；应使用其他适合的角色配置。Thinking 可声明预算或 `thinking_level`，不能同时设置两者。仓库不自动给 Gemini 模型猜测映射。

vLLM 示例：

```yaml
metadata:
  v2_reasoning_modes:
    instruct:
      chat_template_kwargs:
        enable_thinking: false
    thinking:
      chat_template_kwargs:
        enable_thinking: true
```

该控制进入实际 API payload 的 `extra_body.chat_template_kwargs`，只在审计 metadata 写参数不能代替它。服务端还须加载支持该模板参数的模型。未声明映射、冲突的 `generation.extra`、未知控制字段均拒绝。

## 配置错误处理

未知 operation、未知字段、未知 Experience/Skill option、错误布尔类型和不一致的奖励组批配置均拒绝。容量和已有顶层参数只有一个配置入口：

| 应使用的顶层 settings | 不允许的重复嵌套入口 |
|---|---|
| `experience_capacity` | `experience_options.capacity` |
| `experience_top_k_per_subtask` | `experience_options.top_k_per_aspect` |
| `skill_capacity` | `skill_options.capacity` |
| `evolution_batch_trajectories` | `skill_options.batch_trajectories` |
| `skill_candidates` | `skill_options.candidates_per_target` |
| `ppo_epsilon` | `skill_options.ppo_clip_epsilon` |
| `ppo_positive_margin` | `skill_options.acceptance_gain_margin` |

所有消融开关放入 `ablations`，不能藏在组件 option 中。改变 rollout 数或启用消融须提供对应的 `experiment_name`。

## 验证边界

`tests/test_operation_profiles_v2.py` 通过本地 provider adapter 和安装的 Google GenAI SDK 检查实际 payload，不访问远程 endpoint。`tests/test_exact_action_target_v2.py` 检查原 token 片段与 prefix 评分，并调用安装版 Qwen3.5 的 `get_rope_index` 验证图像 token 的位置保持不变、续写 token 的 `mm_token_type_ids=0`。真实 GPU/endpoint 验证应另看对应运行日志，不能从这些 CPU 测试推导实验准确率。

## Responses 请求边界（二次复审）

完整审计 metadata 留在本地 request journal；发给 Responses 的仅为固定白名单标量索引及完整 metadata 摘要，最多 16 项、key 不超过 64 字符、value 不超过 512 字符。长索引值改发摘要，嵌套审计对象不发送。[官方 metadata 约束](https://developers.openai.com/api/reference/python/resources/responses/methods/create)

GPT-5.4（含日期快照名）在非显式 `none` reasoning 时省略 temperature/top_p；显式请求不支持的 logprobs 则拒绝。其他模型可通过 `metadata.sampling_requires_reasoning_none` 显式采用这一策略，不能由模型家族名字推定兼容性。requested/effective sampling 字段保存到 provider raw、知识验证记录与 usage；effective=null 表示未发送，不猜测服务器默认值。`generation.extra` 不允许绕过受管参数。这里适配的是请求契约，未据此宣称实际端点已通过。[GPT-5.4 参数兼容性](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4)

远程 response 返回的实际 `model`、`model_version`/`modelVersion` 与 `system_fingerprint`（若有）记录于 usage；它们不能替代不可访问的远程权重身份，也不保证服务未来版本稳定。

## 各本地角色的资源身份

`model_resources` 将 executor、scorer、knowledge_builder 指向各自资源清单，共享路径只哈希一次。清单覆盖实际模型与 processor/tokenizer 目录（包括 chat template）、revision 和权重 shard；preflight 标记 `pending_execute`，execute 在任何模型/API 调用之前计算所有本地角色的权重哈希并升级为 `complete`。Runtime 在建 journal 前检查角色、辅助文件内容与权重清单/stat，重新启动 execute 会重新计算权重内容哈希。资源或配置变化要求新的运行身份，不能混合复用旧结果。

仅直接构造的无资源绑定脚本化 Runtime 标记 `unbound_programmatic_runtime`，不可当作正式可复现身份。远程模型无法冻结服务端权重，其复现范围依赖配置和可返回的版本信息。
