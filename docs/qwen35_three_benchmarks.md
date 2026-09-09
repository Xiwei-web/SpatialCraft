# Qwen3.5-9B 三数据集实验准备与恢复说明

## 当前状态（2026-09-07）

**正式 SpatialCraft 实验尚未启动，没有正式 accuracy、skill 或 experience 结果。**
本轮已按用户新确认协议接通 agent → 工具反馈 → 知识更新 → NP-PPO → 冻结部署链路，并通过离线接线/恢复测试。
**最新协议、参数与正式入口请优先阅读 [confirmed_protocol.md](confirmed_protocol.md)。** 本文保留固定数据划分和早期 smoke 的来源说明。
早期真实 provider 记录在 `spatialcraftLog/smoke/`，最新验证在 `spatialcraftLog/validation/protocol-20260907/`。尚未调用 OpenAI API 或运行真实全工具组合。

日志根目录：`/l/users/xiwei.liu/spatialcraftLog`。
准备目录：`preparation/qwen35_9b_seed42_v1/`。

## 固定数据划分

| 数据集 | environment（积累） | deployment（冻结评估） | 分层字段 |
|---|---:|---:|---|
| RoboSpatial | 175 | 175 | compatibility / configuration / context |
| ERQA | 200 | 200 | 原始八类 question_type |
| Omni3D | 251 | 250 | 原始 answer_type：float / int / str |

依据本地草稿实验划分和 SMA 附录 C.2：seed=42，类别内 50/50，奇数余数交替分配且 environment 优先。
论文没有完整给出随机数算法、类别遍历顺序和最终 split IDs，因此本实现额外固定为：
类别按名称排序、类别内按 task_id 排序、每个数据集一个 Python `random.Random(42)` 顺序 shuffle。
**不声称与 SMA 作者未提供的 split IDs 逐条相同。**

这是 task-instance 划分，不是 image-disjoint 划分。两侧 task_id 没有交集；共享图像数量分别为
RoboSpatial 40、ERQA 25、Omni3D 88。按 question+image hashes+choices 判定，RoboSpatial 和 Omni3D
各有 1 组跨 split 的相同内容，ERQA 为 0。该源数据重复现象已记录，尚未擅自去重或改成按图像分组；
若选择更严格的去重/分组协议，必须新建划分版本，不能与当前实例级 50/50 协议混用。

每个数据集目录：

- `splits/environment.jsonl`、`splits/deployment.jsonl`：统一 TaskSample 的公开输入；reference_answer=null，去除 GT 掩码、深度及非必要元数据。
- `verification/environment.jsonl`、`verification/deployment.jsonl`：评估器专用完整 TaskSample。目录 0700、文件 0600；普通 executor 不能从这里组装提示词。
- `rollouts/`、`experiences/`、`skills/`、`ppo/`、`artifacts/`、`checkpoints/`、`errors/`、`results/`：预留目录，**目前为空**。
- 根 `manifest.json`：源 Parquet/PDF/代码/模型配置哈希、样本数、类别数、输入文件哈希、重叠统计和未确定参数。没有再次重算整个模型权重哈希。

公开输入和评估标签是访问约定及文件分离，不是针对同一 Unix 用户的安全隔离。后续运行器必须显式控制谁能读标签。

## 准备命令

```bash
conda activate spatialcraft
cd /home/xiwei.liu/spatialcraft
export PYTHONPATH="$PWD/src"
python -m spatialcraft.experiments.prepare \
  --output /l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1
```

重复执行会复核输入，不重写一致的文件。若绑定的代码、数据、模型配置或种子变化，会拒绝覆盖；
使用新的输出目录生成新版本。完成前中断可再次执行相同命令，已有一致文件会复用。
注意：本轮已修改代码，不能在旧 v1 目录重新运行 prepare。新 `experiments.run` 直接复用其固定数据并另行绑定当前代码，不要求重建 preparation。

## 阶段级断点机制

`experiments/journal.py` 的 `RunJournal` 将一次模型调用等工作视为独立阶段：

```text
<journal>/journal.json                  # 不可变运行绑定及哈希
<journal>/stages/<dataset>/<stage>/
  inputs.json                          # 本阶段不可变输入
  attempts/000001.json                  # running/interrupted/failed/completed
  result.json                          # 原子提交的结果及校验和
```

有完整 `result.json` 时校验后直接复用；没有提交的阶段重新执行。输入或模型/协议绑定不一致则拒绝恢复。
单阶段使用进程锁，进程退出后内核自动释放。已测试人为 `KeyboardInterrupt`、输入变化和结果篡改。
记录保存错误类别而非任意异常内容，以降低凭据写入日志的风险；完整控制台日志由调用方保存。

这是阶段级 at-least-once 重试，不是外部调用 exactly-once：如果外部调用已经执行但结果尚未落盘即崩溃，
该调用可能重复。知识更新必须先生成完整新快照，再原子发布，不能让阶段回调原地修改共享知识库。
本轮已接入完整 rollout/tool/knowledge/PPO 运行器，并通过模拟模型下的中断、恢复、完成阶段复用测试；真实空间工具组合仍待小规模验收。

## 真实 provider smoke（不是正式结果）

脚本：`scripts/inference/smoke_prepared_benchmarks.py`。
仅使用 environment 的公开输入，依次选择 RoboSpatial 首题、ERQA 图片数最多的一题、Omni3D 首题。
不读取 deployment/verification，不调用工具，不生成知识，不计算 reward。
使用 BF16、temperature=0、64 个输出 token、关闭 thinking，图像像素预算 65536–262144。
这些是短时诊断参数，不能冒充论文中的正式生成/图像设置。

在已分配的 GPU Slurm 作业内执行（替换为实际 job ID，不自动申请或取消用户的 allocation）：

```bash
srun --jobid=<your-job-id> --overlap --nodes=1 --ntasks=1 --time=00:15:00 \
  env PYTHONPATH=/home/xiwei.liu/spatialcraft/src HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
  /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python \
  /home/xiwei.liu/spatialcraft/scripts/inference/smoke_prepared_benchmarks.py \
  --preparation /l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1 \
  --journal /l/users/xiwei.liu/spatialcraftLog/smoke/qwen35_9b_provider_v2
```

首次加 `--interrupt-before erqa` 会在 RoboSpatial 提交后主动退出（75），随后去掉该选项即可恢复。
`v1` 是修正脚本回调绑定前的一次验证记录，保留供审计；当前脚本使用 `v2`，不能强行覆盖旧版本绑定。

实际验证已完成：作业 213110 内的 A100 40GB 上，v2 的 RoboSpatial（1 图）、ERQA（16 图）、
Omni3D（1 图）均生成并原子提交。输入/输出 token 数依次为 285/4、3809/64、300/6。
ERQA 命中 64-token 上限，输出不完整，因此只能证明多图调用正常，不能视为有效评估答案。
首次在 ERQA 前人为退出，第二次复用 RoboSpatial 后完成余下两项；第三次三个结果全部复用，
没有再次生成或加载权重。尝试状态分别为 `[completed]`、`[interrupted, completed]`、`[completed]`；
已重新校验三个提交的 SHA-256。日志见 `console-first.log`、`console-resume.log`、`console-reuse.log`。
验证作业步骤已退出，未保留模型服务；用户原有交互式 allocation 未取消。该早期阶段单元测试为 36 passed；旧协议 v1 为 62 passed，新 v2 结果见 `validation/protocol-v2-20260907/`。
旧 smoke 绑定旧代码，当前代码不得强行续写 v2。最新 provider 检查脚本是 `scripts/inference/check_protocol_provider.py`。

## 最新正式启动前检查

1. 最新 v2：1 pass、4 rollouts/task、top3/subtask、每 parent 六条相关轨迹、每 task 屏障最多两 parents、三候选、容量100/20；50/8 steps、1024 tokens、训练temperature0.7、thinking=true。旧 batch8/nonthinking 默认不再适用。
2. 新入口中 Qwen9 负责执行/知识构建/严格 PPO；`text-embedding-3-small` 负责两分支向量。待用户设置 `OPENAI_API_KEY` 后验证连接，本轮不调用付费接口。
3. 使用 `experiments.run` 的严格运行器，不能用旧 generic pipeline 的示例默认值冒充正式方法。多模态动作评分和原生工具解析已在真实 Qwen9 上验证。
4. 单次 deployment overall/分类 accuracy 已接入；数值容差、pointing 和 Omni3D 答案格式细节仍需与目标比较协议核对，不称作已复刻官方 evaluator。
5. 工具按需加载并在串行调用后释放重型运行时，尚需真实小规模组合测试；单卡长上下文、多图和 SAM3/MoGe 等负载不因各自接口存在就算通过。

正式部署前冻结知识快照；deployment 不能更新 experience/skill/可靠性统计，也不能用 deployment 得分选择超参数或快照。
后续设置 key、确认工程参数并通过真实小规模故障恢复验证，再依次完整运行三个数据集。
