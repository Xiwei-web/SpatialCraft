# MemP 空间任务基线

这是 MemP 的 **paper-aligned offline spatial adaptation**：将原论文的离线过程记忆机制接入本项目空间任务与真实工具执行器。它不等同于 ALFWorld / TravelPlanner 原论文实验的完整复现，也不把已有 `memp_reflection` 换名作为原方法。

参考来源：

- [Memᵖ: Exploring Agent Procedural Memory，arXiv v4，2026-04-15](https://arxiv.org/html/2508.06433v4)。本文区分离线建库和测试期间更新两种设置。
- [官方仓库固定版本 3066a1b280a39c7433ae37d9a745861903a5d3c4](https://github.com/zjunlp/MemP/tree/3066a1b280a39c7433ae37d9a745861903a5d3c4)。固定引用有助于区分论文机制、公开代码行为与本项目适配。

## 方法与边界

1. 在 preparation 声明的 environment 任务上运行工具 agent，保存轨迹与评估结果。
2. 从成功轨迹构建记忆：保留公开任务、完整可见的 action / tool JSON 语义轨迹，并由 LLM 总结可复用 script。历史图像、张量保存描述符与 SHA，历史 URI 转为 `MEMORY_IMAGE_N` / `MEMORY_ARTIFACT_N` 占位符，不将历史图像重新注入上下文；当前任务的所有图像仍按原顺序、high detail 提供。执行器的请求 ID、token 日志和机器路径保留在原始审计，不应成为示例正文。
3. 以记忆来源的 **source query** 作为 embedding 索引键，对新任务使用 cosine Top-k 检索。
4. 按 `--memory-format` 提供记忆：`trajectory` 为轨迹示例，`script` 为抽象步骤，`proceduralization` 为两者结合，默认使用组合形式。
5. TEST 部署读取冻结记忆。测试结果用于评估，不用于新增、修改或重排记忆。若 environment 没有符合成功门槛的轨迹，则冻结空库，部署阶段不注入记忆，并明确报告 `memory_count=0`；这不代表已经构建出有效记忆。

原论文 §4.2 的离线设置使用成功训练轨迹，比较上述三种表示，并研究 Query、Random 与 AveFact 检索；§4.3 另行研究测试期间的更新。这里实现 Query 检索与离线冻结协议，本轮没有 online 或 AveFact 入口。[论文 §4.2–4.3](https://arxiv.org/html/2508.06433v4)

已有 `spatialcraft.experiments.run_memory_baseline --method memp_reflection` 是通用反思适配：成功或失败轨迹都可能生成新的反思条目，以反思内容检索。原论文的 online Adjustment 则针对导致失败的**已检索旧记忆**进行修订，二者不是同一更新机制。新入口使用独立结果目录，结果应标为 `MemP offline spatial adaptation`。

与无工具 RAG 比较时，工具权限不同会影响准确率、步数和成本，不能将差异全部归因于记忆方法。判断方法贡献时，应匹配数据划分、模型、工具集合、环境采样次数、上下文和输出预算，并分别报告建库与部署成本。

## 默认配置

这些是本项目的运行选择，不是原论文官方默认值。尤其是 `none / 4096` 采用保守初值，不能解释为已经确定了最终实验预算。

| 参数 | 默认值 | 环境变量 |
| --- | --- | --- |
| 数据集 | `robospatial erqa omni3d sat` | `MEMP_DATASETS`，空格分隔 |
| 可选额外数据集 | `viewspatial`，需对应 preparation 支持 | 同上 |
| Agent / script 模型 | `gpt-5.4`，可选 `gpt-5.4-mini` | `MEMP_MODEL` |
| Reasoning effort | `none` | `MEMP_REASONING_EFFORT` |
| 单次最大输出 token | `4096` | `MEMP_MAX_OUTPUT_TOKENS` |
| Temperature | `0` | `MEMP_TEMPERATURE` |
| 每个 environment 任务采样 | `1` | `MEMP_ENVIRONMENT_ROLLOUTS` |
| 每次任务最大步数 | `50` | `MEMP_MAX_STEPS` |
| 检索 Top-k | `3` | `MEMP_TOP_K` |
| 记忆表示 | `proceduralization` | `MEMP_MEMORY_FORMAT` |
| Embedding | `text-embedding-3-large` | `MEMP_EMBEDDING_MODEL` |
| 阶段 | `all` | `MEMP_STAGE` |
| 输出目录 | 必须显式指定 | `MEMP_OUTPUT` 或 `--output` |
| Preparation | `/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1` | `MEMP_PREPARATION` |
| Benchmark 根目录 | `/l/users/xiwei.liu/benchmark` | `MEMP_BENCHMARK_ROOT` |
| API key 文件 | `/home/xiwei.liu/.config/spatialcraft/openai_api_key` | `MEMP_API_KEY_FILE` |
| Python | `/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python` | `MEMP_PYTHON` |

输出目录必须位于源码 checkout 之外，建议使用 `/home/xiwei.liu/spatialcraftRuns/<run_name>`；不要将运行结果写入 `/home/xiwei.liu/spatialcraft` 内部。

Wrapper 最后追加命令行参数，所以显式 CLI 参数可覆盖环境变量默认值。需要更大的输出预算时，可明确指定 `--reasoning-effort medium --max-output-tokens 16384`，并为该配置使用新的输出目录。

官方离线脚本中的示例参数是 `direct` 构建、`query` 检索、Top-10、冷启动前 300 条、每批 10 个任务、最多 30 步；其脚本只注入 workflow。它包含模型占位符和依赖配置，不能把这些示例直接当作本项目可执行的完整复现配置。[官方离线脚本](https://github.com/zjunlp/MemP/blob/3066a1b280a39c7433ae37d9a745861903a5d3c4/ProcedureMem/run_memp_offline.py)

## 运行

默认只做预检，不调用模型、embedding 或空间工具。先检查解析后的参数、数据准备与资源报告：

```bash
cd /home/xiwei.liu/spatialcraft
export MEMP_OUTPUT=/home/xiwei.liu/spatialcraftRuns/memp_gpt54_offline_v1
bash memp/run_memp.sh
```

也可直接传 CLI 参数，例如先预检单个数据集：

```bash
bash memp/run_memp.sh \
  --output /home/xiwei.liu/spatialcraftRuns/memp_mini_robospatial_v1 \
  --datasets robospatial --model gpt-5.4-mini
```

API key 只从文件读取，脚本传递的是路径。不要将密钥正文放进命令行、Slurm 参数或结果说明。

确认配置后，以下命令会提交作业并启用真实执行，包含付费 API 调用：

```bash
export MEMP_OUTPUT=/home/xiwei.liu/spatialcraftRuns/memp_gpt54_offline_v1
sbatch --export=ALL memp/memp.sbatch
```

Slurm 默认 `cscc-users / cscc-gpu-p / cscc-gpu-qos`，1 张 GPU、8 CPU、64 GB 内存、48 小时。GPU 用于本地空间工具，agent 与 script 生成走 API。调整 Slurm 资源时可使用 `sbatch` 参数；更改模型等运行参数时通过上述 `MEMP_*` 环境变量或追加 CLI 参数。脚本本身不会调用 `sbatch`。

需要逐阶段执行时，`environment` 和 `deployment` 在已有 GPU 分配中运行；`build` 可以在 CPU 上执行，但仍校验绑定的工具文件完整性：

```bash
bash memp/run_memp.sh --stage environment --execute
bash memp/run_memp.sh --stage build --execute
bash memp/run_memp.sh --stage deployment --execute
```

`--execute` 是显式执行开关；`memp.sbatch` 已传入该开关。`--stage all` 按 environment → build → deployment 顺序执行。各阶段应使用相同输出目录和方法参数，不能用修改过的配置混入旧运行。

## 阶段、费用与续跑

| 阶段 | 工作 | 主要调用成本 |
| --- | --- | --- |
| `environment` | 采集 environment 工具轨迹并评估成功情况 | Agent API；本地工具计算 |
| `build` | 从成功轨迹提取 script、构建检索库 | Script LLM 与 embedding；具体调用取决于记忆表示 |
| `deployment` | 用冻结库检索并执行 TEST | 查询 embedding、Agent API、本地工具计算 |
| `all` | 顺序执行以上阶段 | 三阶段合计 |

`max_output_tokens` 是单次请求上限，不是全实验总 token 或费用上限；`max_steps` 与任务数量、rollout 数共同决定最坏情况下的调用规模。应分别查看各阶段 journal 和用量报告。

输出绑定输入、运行配置和源代码哈希；续跑时验证这些绑定并拒绝不一致来源。本入口不自动制作代码快照，因此第一次执行后应保持相关源码不变；方法、预算或源码变更使用新的输出目录，保留旧目录供审计。

模型用量与 embedding 缓存持久化到运行产物。续跑只保证已 commit 阶段不会重放；若网络请求已经完成、进程在 commit 前退出，该次调用可能在续跑时再次付费，不能声称 exactly-once。

单元测试和预检能够验证协议与调用接线。只有实际完成对应数据集的运行后，才能报告准确率、成本或声称该配置完成实验。

## 产物与验证

每个数据集的运行目录下：

- `memory/frozen.json`：冻结的轨迹、script、向量及 snapshot ID；`sources` 将每条记忆关联到源 trajectory journal、rollout 序号、内容哈希和 script 阶段，不把这些来源审计字段注入模型。
- `memory/build_summary.json`：环境成功数、记忆数、已提交 script 调用数与用量。
- `progress/environment.json`、`progress/deployment.json`：各阶段进度与 actor 用量。
- `results/deployment.json`、`results/predictions.jsonl`：部署指标与逐题结果。
- `stages/`：可续跑的模型、工具、检索及构建记录；`embedding_cache/usage/`：embedding API 用量记录。缺失的 token 字段保持未知，不当作零用量。

2026-09-14 离线验证：以下测试 **122 项通过**，使用模拟 provider 和工具，不产生 API/GPU 推理调用：

```bash
PYTHONPATH=src:. /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python -m pytest \
  memp/tests tests/test_react_api_baseline.py \
  tests/test_action_recovery.py tests/test_api_baseline.py -q
```

真实 RoboSpatial 数据预检也通过：environment 175、deployment 175，11 个工具 schema，资源路径无缺失。该检查验证数据配对、图像哈希和工具路径；真实工具推理与在线模型调用仍需在实际作业中验证。临时预检目录只用于诊断，正式实验请新建运行目录。

使用 medium 等推理模式时，正常执行、script 构建、动作恢复和强制收尾调用均采用配置的 max_output_tokens，包含推理 token；所有调用省略 temperature/top_p。none 模式保留原有恢复 1024、收尾 512 的短调用预算。每题工具步数上限仍由 max_steps 控制。
