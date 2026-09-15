# MemRL-R / MemRL-GT 空间任务基线

本目录将 MemRL 的记忆效用学习与双阶段检索接入 SpatialCraft 的空间工具 agent，并提供 SMA 定义的两种环境反馈变体。结果应标为 **MemRL-R / MemRL-GT spatial adaptation**，不等同于原 MemRL 在 ALFWorld、BigCodeBench 等数据集上的原样复现。

参考来源：

- [MemRL 原论文，arXiv:2601.03192v2](https://arxiv.org/html/2601.03192v2)，方法见 §4，超参数见附录 E，训练后冻结评估见 §5.2。
- [MemRL 官方代码固定版本 c1b322ca43de36ddf64c6712f89d0095bfc35ce0](https://github.com/MemTensor/MemRL/tree/c1b322ca43de36ddf64c6712f89d0095bfc35ce0)。本目录独立实现论文机制，复用本项目的数据验证、API 执行器和工具协议。
- [Spatial Memory Agent，附录 C.4.4–C.4.5](https://arxiv.org/html/2608.12743v1#A3.SS4)。`R` / `GT` 是这里采用的空间任务反馈变体命名；原 MemRL 论文未将其作为两个独立方法命名。

## R 与 GT 的区别

| 变体 | 环境阶段 LLM 反思获得的信息 | actor、检索与部署 |
| --- | --- | --- |
| `R` | 公开任务图文、agent 执行轨迹和输出、标量奖励 | 当前任务始终不提供标准答案 |
| `GT` | 与 R 相同，额外提供当前环境任务经验证的 `reference_answer` | 额外答案仅供环境反思；部署阶段不生成反思 |

两者都通过环境任务评估获得奖励并更新记忆效用。本空间适配采用正确为 1、错误或受控执行失败为 0 的二元奖励；它与原 MemRL 部分官方配置的成功 +1、失败 −1 取值不同。`GT` 的区别是反思模型获得更丰富的纠错信息，并非部署时读取测试标准答案。环境学习产生的 GT 反思可进入冻结记忆，因此 GT 属于使用额外环境监督的对照。

本实现采用 SMA 的反思文本适配：反思模型读取当前任务图像与完整可见语义轨迹，检索后注入 actor 的记忆正文只包含来源公开 query、choices 和 LLM reflection，不附加历史原始轨迹或历史图像。原 MemRL 官方默认成功记忆包含 script＋trajectory、失败记忆包含 reflection＋failed trajectory；这里没有宣称复现其完整 `proceduralization` 表示。[官方经验构建](https://github.com/MemTensor/MemRL/blob/c1b322ca43de36ddf64c6712f89d0095bfc35ce0/memrl/service/builders.py)

同一个 `--variant both` 运行会共享经校验的数据文件，但 R 与 GT 各自执行环境任务，分别维护记忆、Q 值、模型调用 journal 和 embedding 缓存；不复用另一变体的采样轨迹或记忆。

## 学习与检索流程

每个环境任务依次完成：

1. 用当前公开 query 的 embedding，从已有记忆中检索相关经验。
2. 将选中经验注入空间工具 agent，运行当前任务并获得标量奖励。
3. 对实际注入上下文的旧记忆更新 `Q ← Q + learning_rate × (reward − Q)`。
4. 调用 LLM 对本次轨迹进行反思，写入新的 intent–experience–utility 条目。新条目的 Q 默认是 0，不将自身任务奖励当作该经验的迁移效果。
5. 下一任务可以检索前面任务积累的经验。完成所有环境轮次后冻结最终记忆，部署阶段只读取。

成功和失败轨迹均参与学习；基础设施故障会抛出错误，不能伪装成模型答错后继续建库。这一循环与 MemP 的“先采集环境轨迹，再统一从成功轨迹建库”不同。学习不修改 backbone 权重。

双阶段检索先保留 cosine 相似度严格大于 `similarity_threshold` 的 Top-`candidate_k` 条目，再按以下分数取 Top-`top_k`：

```text
score = (1 − utility_weight) × zscore(similarity)
        + utility_weight × zscore(Q)
```

本适配按候选条目做 z-score，没有相关候选时不注入记忆。默认阈值通过**公开 environment query** 的 embedding 两两 cosine 的 80% 分位点校准，校准在显式执行时发生；部署数据与部署标签均不参与阈值选择。也可通过 `--similarity-threshold` 指定固定阈值。

原文与官方代码有实现细节差别：官方 `retrieve_query` 先取相似 query clusters 再展开经验，similarity 使用数据集预计算统计，Q 使用候选统计并截断。本目录选择论文描述的条目级双阶段检索与候选池标准化；不移植旧版仅按 Q 排序的 `ValueAwareSelector`。[官方实际检索实现](https://github.com/MemTensor/MemRL/blob/c1b322ca43de36ddf64c6712f89d0095bfc35ce0/memrl/service/memory_service.py)

原论文使用不同数据集的 `5/3` 或 `10/5` 检索组合，并报告 10 轮 runtime 学习。本适配默认 `candidate_k=10, top_k=3`，环境单轮、每题一次采样，以匹配本项目已有实验预算。两者不能混称为原论文默认参数。更改 `--environment-passes` 可以进行多轮环境学习；使用最后一轮的冻结库，不能根据部署准确率挑选轮次或 checkpoint。

## 默认参数

| 参数 | 默认值 | Bash 环境变量 |
| --- | --- | --- |
| 变体 | **必须指定** `R` / `GT` / `both` | `MEMRL_VARIANT` |
| 数据集 | `robospatial erqa omni3d sat`，也支持 `viewspatial` | `MEMRL_DATASETS`，空格分隔 |
| actor / 反思模型 | `gpt-5.4-mini`，可选 `gpt-5.4` | `MEMRL_MODEL` |
| reasoning effort | `medium` | `MEMRL_REASONING_EFFORT` |
| 单次最大输出 token | `16384` | `MEMRL_MAX_OUTPUT_TOKENS` |
| temperature | `0`，仅 `none` 推理模式发送 | `MEMRL_TEMPERATURE` |
| 环境任务轮次 | `1` | `MEMRL_ENVIRONMENT_PASSES` |
| 每轮每个环境任务采样数 | `1` | `MEMRL_ENVIRONMENT_ROLLOUTS` |
| 每次任务最大步数 | `50` | `MEMRL_MAX_STEPS` |
| 相似候选数 / 最终注入数 | `10` / `3` | `MEMRL_CANDIDATE_K` / `MEMRL_TOP_K` |
| Q 分数权重 | `0.5` | `MEMRL_UTILITY_WEIGHT` |
| Q 学习率 | `0.3` | `MEMRL_LEARNING_RATE` |
| 新记忆 Q | `0` | `MEMRL_Q_INIT` |
| 固定相似度阈值 | 未指定，默认自动校准 | `MEMRL_SIMILARITY_THRESHOLD` |
| 校准分位点 | `0.8` | `MEMRL_THRESHOLD_QUANTILE` |
| Embedding | `text-embedding-3-large` | `MEMRL_EMBEDDING_MODEL` |
| 阶段 | `all`，可选 `environment` / `deployment` | `MEMRL_STAGE` |
| 输出目录 | **必须指定，位于源码目录外** | `MEMRL_OUTPUT` |

数据和运行环境默认沿用已有 MemP 基线：

- Preparation：`/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1`，由 `MEMRL_PREPARATION` 覆盖。
- Benchmark：`/l/users/xiwei.liu/benchmark`，由 `MEMRL_BENCHMARK_ROOT` 覆盖。
- Python：`/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python`，由 `MEMRL_PYTHON` 覆盖。
- API key 文件：`/home/xiwei.liu/.config/spatialcraft/openai_api_key`，由 `MEMRL_API_KEY_FILE` 覆盖；不将密钥正文放入命令行或 Slurm 参数。
- 本地空间工具：使用项目的 `SpatialToolPaths` 与 `SPATIALCRAFT_TOOL_ROOT` 配置。

CLI 参数最后追加，因此可覆盖 wrapper 中的环境变量。`16384` 是**单次模型请求**的输出上限，包含推理 token，并非全任务或全实验费用上限。环境执行、反思、部署与 embedding 都应计入实验成本。

## 预检和执行

默认只验证数据、参数、图像哈希与工具路径，并写出 `preflight.json`。不读取密钥、不调用 LLM/embedding，也不加载 GPU 模型：

```bash
cd /home/xiwei.liu/spatialcraft
bash MemRL/run_memrl.sh --variant both \
  --output /home/xiwei.liu/spatialcraftRuns/memrl_mini_medium16384_foursets_v1
```

仅准备 ViewSpatial 或单个变体时：

```bash
bash MemRL/run_memrl.sh --variant R --datasets viewspatial \
  --output /home/xiwei.liu/spatialcraftRuns/memrl_R_viewspatial_v1
```

已获得 GPU 分配时，追加 `--execute` 启用真实 API 与工具调用。逐阶段使用同一输出目录与变体选择：

```bash
export MEMRL_VARIANT=both
export MEMRL_OUTPUT=/home/xiwei.liu/spatialcraftRuns/memrl_mini_medium16384_foursets_v1
bash MemRL/run_memrl.sh --stage environment --execute
bash MemRL/run_memrl.sh --stage deployment --execute
```

`all` 会完成环境学习和冻结部署，没有独立的 `build` 阶段。R 与 GT 顺序执行，各自的配置、采样和产物独立。

以下命令会提交 Slurm 作业并启用真实执行；本轮生成脚本本身没有提交作业：

```bash
export MEMRL_VARIANT=both
export MEMRL_OUTPUT=/home/xiwei.liu/spatialcraftRuns/memrl_mini_medium16384_foursets_v1
sbatch --export=ALL MemRL/memrl.sbatch
```

Slurm 默认账号 `cscc-users`、分区 `cscc-gpu-p`、QOS `cscc-gpu-qos`，1 张 A100 SXM4 40GB、8 CPU、64 GB、48 小时，排除 `gpu-05,gpu-[49-56]`。GPU 用于本地空间工具，actor 和反思模型通过 API 调用。`both` 会在同一个作业里顺序运行两份实验，所需时间及费用相应增加。

`MEMRL_PROJECT_ROOT` 可指向完整代码快照，用于指定 batch script 实际启动的源码。脚本不自动制作快照；在首次执行后应保持源码和参数不变。只查看参数可运行 `bash MemRL/run_memrl.sh --help`。

## 结果、冻结与续跑

`<output>/data/` 存放共享的校验后数据；`<output>/R/<dataset>/` 与 `<output>/GT/<dataset>/` 存放各自记忆、进度、部署结果和调用记录。总体状态写入 `<output>/results/<stage>_summary.json`。保留 `preflight.json` 和 `run_identity/` 可以核对变体、参数、输入与源码来源。

运行身份包括 `MemRL/`、复用的 `memp/`、`RAG/data.py`、`src/`、配置和提示词文件。相同输出目录不能切换变体选择、模型、采样预算、数据或源码；单阶段与 `all` 可以在同一身份下续跑。修改设置后使用新的输出目录。

部署只读取环境阶段冻结的最终记忆，测试答案仅由评分器用于汇报指标；不据测试奖励更新 Q、生成反思、写入经验或调整检索阈值。相同任务或相同图文内容的来源不能作为当前任务的可检索示例。历史媒体路径仅保留为经验描述，不能授权当前工具访问其他任务文件。

模型调用和学习记录持久化以支持恢复。已经提交的记录可以重放恢复状态；若 API 已返回而进程在持久化前中断，续跑仍可能重复付费，不保证外部 API exactly-once。部署最终准确率以整个部署集为分母，包括已处理但未生成有效答案的受控失败；运行中进度需结合已处理数量与总题数阅读。预检和模拟测试不能替代真实实验结果。

每个变体、数据集目录内的主要产物：

- `memory/frozen.json`：最终记忆条目、向量、Q 值、阈值校准信息和快照哈希。
- `results/environment.json`：环境正确率、actor 用量、反思调用与用量、效用更新数。
- `results/deployment.json`、`results/predictions.jsonl`：冻结部署指标及逐题结果。
- `progress/environment.json`、`progress/deployment.json`：已处理数、总题数和当前进度。
- `stages/`：校准、逐题检索、执行、反思与记忆更新的 journal；环境更新记录保存旧条目 Q 更新及新条目来源，可审计重放。
- `embedding_cache/usage/`：embedding 调用用量；复用缓存不代表产生新的外部调用。
