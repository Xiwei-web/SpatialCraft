# 输出超限收尾策略（2026-09-09）

当前工作区的trajectory预算策略以 `docs/action_recovery_v2.md` 为准：token截断使用action recovery，工具步数耗尽才forced-final。本文中旧输出超限收尾描述和已提交快照记录属于历史协议。

用户确认：达到单次 4096 tokens 上限时强制收尾并继续，不再因该情况中断整个实验。

实现策略 `force_completion_v1`：

- 正常调用及其提示词不变；仅 `finish_reason=length` 触发恢复。
- Agent：基于原问题、图像和已有工具证据追加一次调用，移除可调用工具，要求直接给简短最终答案。成功后走正常 verifier，结束当前 rollout；随后按原协议继续下一个 rollout/task，不跳过同题应执行的四条轨迹。
- Summary/critique/Skill generation 等：追加一次紧凑重生成，要求保留必要字段、合并重复证据，不接纳截断草稿为知识。
- 每个调用的输出上限仍为原配置（当前 4096），收尾 temperature=0；thinking 开关不变（当前 false）。收尾是**额外一次调用**，因此原调用加收尾的总输出可能超过 4096；全部实际 token 用量仍记账。
- 最多一次恢复，不无限重试。若仍超限、为空或返回工具调用：Agent 记录 truncated、无最终答案、reward=0，然后继续；辅助操作跳过本次更新。Experience Bank/Skill Pool 保持原状，演化队列保留新旧未消费轨迹。检索失败时不注入 Experience；termination 辅助判断失败时结束当前 Skill，随后重新选择，不结束整条 trajectory。
- 原始与恢复模型调用分开保存。恢复请求具有独立 metadata/cache key；成功收尾 transition 保存实际收尾 request，供 PPO 使用。跳过的更新记录 `output_budget_recovery.status=skipped_output_limit`，进度脚本单独统计，不能把这些记录视为成功学到知识。
- CUDA OOM、网络/API、磁盘、数据损坏等基础设施错误仍传播，不伪装为答案或知识。

代码：`src/spatialcraft/experiments/output_budget.py`、`learning.py`、`rollout.py`。
测试：`tests/test_output_budget_recovery.py`，包括 journal 重放和跨题继续。

## 当前生效范围

已于 2026-09-09 提交 **220852**，提交后查询为 `PENDING / Priority`。使用新快照 `code_revisions/output_budget_v1`，仅从上一快照增改三个文件：`learning.py`、`rollout.py`、`output_budget.py`，已登记 code patch 链。旧快照未覆盖，372 条训练轨迹、92 次 Experience 更新及92轮演化保留；从第93题的 Experience 更新恢复。

新入口：`scripts/inference/robospatial_budget.sbatch` → `run_robospatial_budget.sh`。资源为同节点2×A100 40GB、8 CPU、96GB主机内存、48小时，FLA隔离依赖不变。**不要用旧 `run_robospatial_fla.sh` 恢复新版本**。

实际快照115项适用回归通过。完整跨分支检查中的3项失败均属于未纳入此快照的后续 Omni3D/27B 接口，原报告保留；本次没有合入或修改独立Omni3D实验。验证报告位于 `/l/users/xiwei.liu/spatialcraftLog/validation/output-budget-20260909/`。

激活前核验15,660个已提交stage；48,992份历史JSON的校验清单存入运行目录 `code_revisions/output_budget_v1-before.json`，旧指针备份为 `output_budget_v1-parent_patch.json`。日志继续写入原运行目录 `launcher_logs/slurm-220852.log` 和新生成的 `*-220852-budget.log`。

已有失败调用可从 journal 缓存读取，新增收尾调用使用独立 key；已完成的 rollout 和知识事务保持不变。本次验证为离线回归，不代表真实 Qwen 的强制收尾已经实测成功。
