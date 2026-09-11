# acdcaec 二次复审处理说明

日期：2026-09-11。审计基线：`acdcaec583aa0e60ac7483c372d45b5ef1ee8c36`，分支 `thrid`。本说明对应基于该提交的工作区修复，尚未创建新提交。用户原审计是核查依据，其中的建议逐项按实际调用链确认，不作为无需验证的事实。

## 核查结论与代码修复

R01–R08 都存在相应实现缺口，修改方向均合理。以下把工程修复与真实模型实验区分记录。

| 项目 | 核查与修复 | 主要实现 / 测试 |
|---|---|---|
| R01 Skill 非法输出使恢复卡住 | 一次修正仍非法后，诊断提交 `diagnosis_unavailable/is_related=null`，不算作“不相关”、不进入候选队列或质量统计；聚合失败把该批六轨迹归档为 `aggregation_unavailable`，保留失败证据并让后续批次继续；LLM 去重失败保留已接受的合法更新，容量管理仍独立执行。只处理 `KnowledgeValidationError`，基础设施异常继续传播。 | `knowledge/skill/learning_v2.py`；`tests/test_skill_validation_runtime_v2.py` 六种正式 Runtime 中断恢复场景。 |
| R02 Responses 请求不兼容 | API metadata 限定为标量索引白名单与审计摘要，完整信息留在本地；GPT-5.4 非显式 none reasoning 省略 temperature/top_p，requested/effective 参数记录到 provider、知识验证与 usage。extra 不能覆盖受管字段。 | `models/providers/openai_responses.py`、`experiments/knowledge_generator.py`、`experiments/usage.py`；`tests/test_review_usage_responses.py` 经过真实 KnowledgeGenerator + AuditedProvider + Responses adapter，使用假的网络 client。 |
| R03 RAG 示例受 raw audit 挤占 | 专用语义表示保留公开任务、动作参数、工具观察与最终回答，排除 token IDs/raw request 等审计字段；正式 Runtime 用执行器 tokenizer 控制完整示例预算，默认 1024 tokens；通用无 tokenizer 回退明确记为 256 words。记录成功轨迹、拒绝、保存、库内可用数与部署命中。 | `knowledge/evidence.py`、`experiments/baseline_memory.py`；`tests/test_memory_baselines_v2.py` 验证四条带原始审计的短轨迹保存四条且部署命中。 |
| R04 Skill 诊断缺少前置空间信息 | 补充激活入口状态、实际注入经验、由当前动作 artifact 输入追溯的前置工具/图像依赖；无命名依赖时保留最近两条工具观察/空间证据。前置操作只供理解，不纳入该段 evidence_refs、收益归因或固定目标评分。 | `knowledge/skill/learning_v2.py`、`prompts/skill/v2_semantic_gradient.txt`；`tests/test_skill_v2.py`。 |
| R05 几何坐标与数值单位混淆 | source/result frame、coordinate length unit、value unit 分离；投影使用独立像素坐标系，注明源图/处理图像空间及映射；角度不会把 world 长度单位变成 degree。点集 artifact 传递 scale_status，projection/backprojection 契约可组合。 | `tools/builtin/geometry.py`、`tools/spatial_arrays.py`；`tests/test_geometry_contract_r05.py`；[工具矩阵](tool_api_matrix.md)。 |
| R06 Skill-Pro 容量声明与执行不一致 | Runtime 构造前解析唯一有效容量，并检查 settings、descriptor、已绑定 evolver 一致；固定六种子协议拒绝容量小于 6。 | `experiments/run_memory_baseline.py`、`experiments/baseline_memory.py`；容量 10 的测试实际把超容池修剪到 10。 |
| R07 total_tokens 与 latency 范围混淆 | 报表升级 schema 3；显式拆分 generation、embedding、scoring。总量定义为 generation_total + embedding_input + scoring_prefix + scoring_target，缺字段传播 unknown；缓存不重复计费。完整调用时间与 provider 内部计时分别保存，历史缺字段保留 unknown。 | `evaluation/cost_report_v2.py`、`experiments/usage.py`、`experiments/runtime_v2.py`；手算例子为 30+5+43=78，reasoning 不重复计入。 |
| R08 独立本地 KB 身份遗漏 | 统一解析所有本地角色的权重、tokenizer、processor/chat-template、revision；共享文件去重哈希，角色引用资源身份。preflight 明确权重待绑定，execute 在推理前补齐；远程响应如有 model/fingerprint 则保存，但不冒充远程权重冻结。 | `experiments/model_resources.py` 与主入口、基线/冻结迁移共享资源绑定；使用微型假资源测试，不读大模型权重来充当 CPU 测试。 |

额外修复了恢复元数据漂移：首次执行检索/选择时会修改 embedding operation scope，而命中阶段缓存时跳过修改，后续 executor request 因此不同。`LearningBuildersV2.retrieve/applicable` 现在使用临时 scope 并在退出时恢复，六种 Runtime 测试覆盖执行阶段和学习阶段中断后恢复。

Responses 规则依据：[官方 metadata 限制](https://developers.openai.com/api/reference/python/resources/responses/methods/create)、[GPT-5.4 参数兼容性](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4)。这里验证的是本地实际请求构造；没有调用远程端点。

## 方法建议的处理与保留意见

- **5.1 效率结论：同意审计。** Skill 仍由 executor 逐步执行，检索、rewrite、选择和终止都有开销。新报表分项和完整调用时间有助于比较，但调用时间求和不是任务墙钟，跨模型 token 也不等价于美元或 FLOPs。尚不能声称必然降低总在线成本。
- **5.2 因果措辞：同意。** 保留 `task_group_outcome_association_not_causal`；`review.md` 的“成熟负贡献”改为“满足成熟证据要求且平均结果关联为负”。负关联并不是经干预确认的负贡献。
- **5.3 数据隔离：同意，但不静默改写历史划分。** 增加可选严格重复拒绝政策：跨 ID 的相同公开 question/image-content/choices 可被拒绝；共享图像但题目不同单独统计。严格模式发现重复时要求新准备目录/划分版本，不自动重新抽题；旧 manifest 仍采用其原有实例划分，不能据此宣称未见公开输入隔离。新 preparation 使用 `--exact-duplicate-policy reject_cross_split_v1`；默认 `report_only` 不给旧 manifest 追加字段。正式 preflight 重新计算实际图片字节指纹，严格来源的冻结迁移也检查训练题公开内容指纹。
- **5.4 基线复现：同意。** 保留 `faithful_reproduction=false`；六个空间适配只能支撑针对这些实现的比较。独立 `RAG/` 与 `rag_demonstrations` 也不能混为一组结果。
- **5.5 输入上下文：已修正 raw audit 污染。** Summary、Skill 诊断和基线反思/示例共用语义证据表示，原评分/恢复日志仍完整保留。这不是自动分段总结；超长的真实证据仍可能超过上下文，需要根据实际分布设计压缩，不能以增加输出上限代替。
- **5.6 真实闭环：同意，仍未完成。** 两题各四 rollout 未必得到六条合格相关轨迹，也未必触发三个候选和 gate。既有脚本称为“部分闭环 smoke”；完整验收须实际覆盖聚合、三候选、teacher forcing、no-change/拒绝/接受升级、旧队列清理及冻结部署。CPU 测试不代替这些模型实验。

没有需要反对的核心工程建议，但需保留审计中的限定：R05 是元数据契约错误，不是投影公式已被证明错误；R07 是统计口径缺口，不是全部底层用量丢失；R08 的额外缺口涉及独立本地 KB，同实例默认配置原先已有 executor 权重绑定。

## 验证补充

最终完整 CPU 回归：**413 passed，15 warnings，60.78 秒**；无失败或跳过。27 个修改过的 Python 文件 Ruff check 与 format --check 通过，`git diff --check` 通过。警告来自 torch NVML 初始化与 matplotlib/pyparsing 弃用接口。回归过程中发现 metadata 白名单缺失旧 ReAct 恢复所需 `state_id`，已补回该短索引字段并重新运行全部测试。

[可随源码审阅的结果与文件 SHA256](validation/review_acdcaec_cpu.json)；[本地完整 pytest 输出](../artifacts/v2_validation/review_acdcaec_cpu_final/pytest.txt)。摘要位于可纳入版本管理的 docs 目录，原始 artifacts 仍被 git 忽略；没有声称这些文件已经提交或发布。复跑命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python -m pytest -q
```

本轮未运行 GPU、付费 API、完整 benchmark 或跨模型实验，也未产生新的准确率/效率成绩。历史 360 项 CPU 与 GPU/API 组件报告属于审计基线，不能当成本轮新结果。完整真实闭环此前因数据外发及资源费用未明确授权而未运行；本次没有重试提交。
