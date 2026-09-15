# GPT example-retrieval RAG baseline

Environment 阶段保存轻量的历史任务与模型原始输出；deployment 阶段按任务 embedding 的 cosine similarity 检索样例，将历史任务文本和原始输出提供给当前模型。**不生成 reflected summaries、lessons、Experience 或 Skill。**

最初编写时仅做静态检查，未执行实验。2026-09-11 新增 `--reasoning-effort` 并按用户要求准备 GPT-5.4-mini medium 正式运行；本次任务 **229287** 使用medium/16384，状态与验证记录见 [运行说明](runs/20260911_gpt54mini_medium16384.md)。

ViewSpatial同设定任务 **229312** 已单独提交，使用固定2856/2856划分，详见 [ViewSpatial运行说明](runs/20260911_viewspatial_gpt54mini_medium16384.md)。

GPT-5.4四数据集同设定任务 **229347** 已独立提交，medium/16384、从空库建库，详见 [GPT-5.4运行说明](runs/20260911_gpt54_medium16384.md)。

GPT-5.4 ViewSpatial同设定任务 **229405** 已提交，medium/16384、固定2856/2856题；提交时因每用户并发上限排队，详见 [运行说明](runs/20260911_viewspatial_gpt54_medium16384.md)。

GPT-5.4 ViewSpatial **233606**（2026-09-12）接续229405：保留375条记录，未开始的2762题改为各1次，测试2856题各1次；正式输出位于共享home，详见 [变更次数续跑说明](runs/20260912_gpt54_viewspatial_remaining1.md)。

## 方法和默认值

这是独立的**单次视觉问答＋历史示例检索** baseline，不启用空间工具或 ReAct。项目已有 `rag_demonstrations` 是“验证成功的工具轨迹”变体，与本目录定义不同。

| 项目 | 设置 |
|---|---|
| 模型 | `gpt-5.4-mini` 或 `gpt-5.4`，分别建库，使用不同输出目录 |
| Thinking | 默认两阶段 `reasoning_effort=none`；`--reasoning-effort medium` 对环境和部署均生效 |
| 单次生成上限 | 默认4096；`--max-output-tokens 16384` 可覆盖两个阶段；无额外 recovery/forced-final 调用 |
| 环境阶段 | 单 pass；默认每题4次独立视觉问答，不检索历史；temperature=0.7 |
| 部署阶段 | 每题1次；temperature=0；库只读 |
| Embedding | 默认 `text-embedding-3-large`，3072维；可显式选择 small |
| 检索文本 | 公开 question、带标签的 choices、answer_type |
| 检索 | 同数据集 Top-3 不同历史任务，cosine 排序，无阈值 |
| 多 rollout | 全部保存；每题以最早完成且非空的一条输出作为代表，不按正确性选择 |
| 历史样例 | 历史问题、选项、答案类型、原始模型输出；不注入历史图片 |
| 当前视觉输入 | 当前题全部原图，原始字节、原顺序、`detail=high` |
| 评分 | 复用纯模型 baseline 的确定性 verifier，部署 GT 仅进入评分 |

开启 medium 等 thinking 档位时，不发送 `temperature`、`top_p`、`seed`，使用服务端采样设置；运行配置记录实际温度为 `null`，不再宣称环境0.7或部署0。输出上限由 `--max-output-tokens` 决定（包含 reasoning tokens）；本次指定为16384。

环境 rollout 数、温度、Top-k 是可配置默认值，不声称由 RAG 定义唯一确定。可使用 `--environment-rollouts 1` 选择一题一条记录的轻量版本。Responses API 不设置 generation seed；seed42仅用于数据划分/抽样。

全部环境输出保留，包括回答错误、空输出和截断。空输出、截断和意外 tool call 不参与检索；**不使用环境 GT 评分或筛选“正确答案”，不将 GT 替换进历史输出。** 数据适配器读取源记录时可能解析标注，进入 RAG 环境输入前即删除。 原始输出逐字保存，没有反思、压缩、重写、效用重排或容量淘汰。任务 embedding 不包含模型输出或私有标注。

## 数据范围

默认前四个数据集，`--datasets` 可单独选择，也支持 ViewSpatial。

| 数据集 | Environment | Deployment | 来源 |
|---|---:|---:|---|
| RoboSpatial | 175 | 175 | 既有 seed42 固定划分 |
| ERQA | 200 | 200 | 既有 seed42 固定划分 |
| Omni3D | 251 | 250 | 既有 seed42 固定划分 |
| SAT | 300 | 300 | 环境复用提供的准备目录，或按现有 v2 规则从 validation 排序后 seed42 随机抽300题；部署保持 circular test 300题 |
| ViewSpatial | 2856 | 2856 | 复用 `viewspatial_seed42_v1` 半集，不评测全部5712题 |

默认准备目录是 `/l/users/xiwei.liu/spatialcraftLog/preparation/qwen35_9b_seed42_v1`；ViewSpatial 使用同级 `viewspatial_seed42_v1`。SAT 缺少已准备环境输入时，从 `/l/users/xiwei.liu/benchmark/SAT/SAT_val.parquet` 准备，新增环境图片缓存写入本次输出目录。

拒绝跨集合 task ID 重复；对 ID 不同但图片内容/question/choices 完全相同的记录，保留既有划分，在检索当前题时排除完全重复的历史样例，并记录排除的 record ID。真实检查发现 RoboSpatial 和 Omni3D 各有一组这种重复，不能假设既有按 ID 划分已实现内容去重。沿用按题目划分的协议，同一图片上的不同问题可能跨集合；会记录共享图片数，不宣称 image-disjoint。

## 文件结构

```text
RAG/
├── __init__.py
├── run_rag.py             # 两模型共用 CLI，显式 --execute 才调用 API
├── run_gpt54mini.sh       # mini 入口
├── run_gpt54.sh           # GPT-5.4 入口
├── data.py               # 固定划分、标签隔离、媒体校验
├── core.py               # 原始记录、任务向量检索、请求构造
├── runner.py             # 积累、冻结、部署、评分、断点记录
├── continuation.py       # 校验并导入历史调用前缀，减少剩余题采样次数
├── freeze_run.py         # 冻结并核验包含RAG的完整执行源码
├── gpt54mini_medium16384.sbatch # 本次CPU正式任务入口
├── gpt54_viewspatial_remaining1_medium16384.sbatch # 229405变更预算续跑
├── tests/test_rag.py      # 离线检索、隔离、恢复与参数回归
├── tests/test_continuation.py # 前缀保留、拒绝损坏记录、减少次数与续跑回归
└── README.md
```

复用项目的 Responses/embedding provider、数据适配器、RunJournal、原子写入和 verifier；使用已有 `spatialcraft` conda 环境，不需要 GPU 或本地模型权重。

## 命令（仅说明，本次未执行）

以下命令只准备数据、验证输入，不调用 API：

```bash
cd /home/xiwei.liu/spatialcraft
bash RAG/run_gpt54mini.sh --output /l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_rag_v1
bash RAG/run_gpt54.sh --output /l/users/xiwei.liu/spatialcraftLog/runs/gpt54_rag_v1
```

未来执行时，在相同命令后显式添加 `--execute`。两个模型必须使用不同目录。脚本不会自动提交 Slurm，而是在当前进程运行。

也可以分阶段执行（以下命令会调用付费 API，本次未执行）：

```bash
bash RAG/run_gpt54mini.sh --output /l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_rag_v1 --stage environment --execute
bash RAG/run_gpt54mini.sh --output /l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_rag_v1 --stage deployment --execute
```

`--stage deployment` 要求所选各数据集已有完整冻结库，不能从测试题建库。默认 `--stage all` 对每个数据集依次建库和评测，数据集之间不混库。

使用已有 `OPENAI_API_KEY` 环境变量，或仅在执行阶段读取 `--api-key-file`，默认路径 `/home/xiwei.liu/.config/spatialcraft/openai_api_key`。离线准备不读取密钥。`SPATIALCRAFT_PYTHON` 可覆盖 shell 入口的 Python 路径。

## 运行产物与恢复

路径均位于指定的 `<output>`：

| 路径 | 内容 |
|---|---|
| `run_config.json` | 模型、方法配置、代码/配置 hash |
| `preflight.json` | 离线准备报告，不是完成结果 |
| `data/rag_manifest.json` | 固定输入、环境来源与媒体校验 |
| `data/<dataset>/environment.jsonl` | 去标签的环境任务 |
| `data/<dataset>/public.jsonl` | 与原 baseline 一致的部署公开输入 |
| `data/<dataset>/private.jsonl` | 仅供部署 verifier 使用的标签 |
| `<dataset>/environment/records/*.json` | 环境阶段逐条保存的轻量记录 |
| `<dataset>/memory/records.jsonl` | 所有原始 task/output 记录，部署不追加 |
| `<dataset>/memory/index.json` | 任务向量、代表记录ID、embedding身份 |
| `<dataset>/memory/snapshot.json` | 最后提交的冻结标记和文件hash |
| `<dataset>/progress.json` | 实时阶段进度或失败状态 |
| `<dataset>/stages/environment/`、`stages/deployment/` | 模型请求/响应、检索ID与cosine、评分和attempt记录 |
| `<dataset>/results/environment.json` | 建库数量、状态、token用量，不报告环境accuracy |
| `<dataset>/results/deployment.json` | 该数据集accuracy及分类结果 |
| `<dataset>/results/predictions.jsonl` | 每题原始输出、评分与检索来源 |
| `results/accuracy.md` | 部署准确率总表 |
| `results/summary.json` | 最近一次命令所执行阶段的汇总；另存`all/environment/deployment_summary.json` |
| `embedding_cache/usage/` | 单独记录embedding请求与用量 |

相同命令、相同目录可恢复已提交的模型响应与embedding。远端请求成功但本地尚未提交时中断，请求可能重发。改变代码、模型、embedding、Top-k或环境采样设置时须用新目录。部署前后校验冻结库，拒绝混用其他配置或被修改的库。

离线测试可在项目根目录运行 `PYTHONPATH=src:. pytest -q RAG/tests/test_rag.py`；测试使用 fake provider/embeddings，不调用真实 API。

## 改变剩余环境次数的续跑

同配置的续跑继续使用原目录即可。若要减少剩余题的采样次数，必须用新目录，并同时传入`--resume-environment-from ORIGINAL_RUN --resume-completed-calls N --environment-rollouts 1`。`N`表示精确连续的原journal已提交调用数；不依据准确率筛选。每题目标次数为“该题已有调用数”与新预算的较大值，因此既有多次输出全部保留，未开始的题采用新预算。原/新模型、数据、提示词、embedding等设置必须一致。详情和233606的实际配置见上面的续跑说明。
