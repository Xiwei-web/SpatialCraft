# Qwen 推理环境说明

目标环境：`/home/xiwei.liu/miniconda3/envs/spatialcraft`，Python 3.11。
目标模型：`/l/users/xiwei.liu/model/Qwen3.5-9B`、`/l/users/xiwei.liu/model/Qwen3.6-27B`。

## 安装与恢复

关键版本锁定在 `requirements/inference-cu128.txt`，使用官方 PyPI 的 CUDA 12.8 PyTorch wheel。不要混入 CUDA 13 的 PyTorch/vLLM 构建。

```bash
conda activate spatialcraft
cd /home/xiwei.liu/spatialcraft
python -m pip install -r requirements/inference-cu128.txt
```

升级前完整备份位于 `/l/users/xiwei.liu/env_backups/spatialcraft-before-qwen-20260907`。已验证备份能导入 PyTorch 2.4.0+cu121 和 Transformers 4.46.3。需要使用原环境时，可直接激活备份，不要删除或覆盖当前目录：

```bash
conda activate /l/users/xiwei.liu/env_backups/spatialcraft-before-qwen-20260907
```

下载缓存位于 `/l/users/xiwei.liu/env_backups/uv-cache` 和 `pip-cache`，不属于模型权重。

## 离线预处理检查

不加载模型权重，验证 config、tokenizer、chat template、视觉 processor 和图片输入张量：

```bash
conda activate spatialcraft
cd /home/xiwei.liu/spatialcraft
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/inference/smoke_qwen.py --model 9b
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/inference/smoke_qwen.py --model 27b
```

## GPU 推理检查

只能在 Slurm GPU allocation 内运行。脚本遵循 Slurm 的 `CUDA_VISIBLE_DEVICES`，不会指定其他用户的物理 GPU。

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/inference/smoke_qwen.py --model 9b --backend transformers
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 python scripts/inference/smoke_qwen.py --model 9b --backend vllm
```

这两个 smoke test 各自加载模型，串行运行以避免显存冲突。它们向模型输入内存中的红色测试图片，生成最多 32 个 token，不访问真实 benchmark、不调用付费 API。环境级 smoke test 不代表 SpatialCraft 的 provider、工具调用、多模态 PPO 或完整实验已经验证。

## vLLM 服务

在激活 `spatialcraft` 的 GPU 作业中：

```bash
bash scripts/inference/serve_qwen.sh 9b
# 27B 默认需要两个已分配的 GPU；单张大显存卡可显式设置 TP=1。
bash scripts/inference/serve_qwen.sh 27b
```

默认仅监听计算节点的 `127.0.0.1:8000`，启用 `qwen3` reasoning parser、`qwen3_coder` tool parser，保留视觉输入；上下文 8192，最大并发 2，eager 模式。不要在登录节点运行。服务与客户端若不在同一节点，需按集群规则配置 SSH 隧道；不要把计算节点的 localhost 当作登录节点的 localhost。

可覆盖 `SPATIALCRAFT_TP_SIZE`、`SPATIALCRAFT_MODEL_PORT`、`SPATIALCRAFT_MAX_MODEL_LEN`、`SPATIALCRAFT_MAX_NUM_SEQS` 和 `SPATIALCRAFT_GPU_MEMORY_UTILIZATION`。27B 单 A100 40GB 放不下完整 BF16 权重，通常需要至少两张 40GB 卡或一张 80GB 卡，并为 KV cache 与视觉输入留余量。不要把磁盘权重大小直接当作总显存需求。

服务别名分别为 `qwen3.5-9b` 与 `qwen3.6-27b`。项目 27B 配置默认使用 vLLM，9B 默认仍使用本地 Transformers。提供 9B vLLM 启动脚本不会自动更改其 provider 配置。

## 验证记录

升级前项目测试：25 passed。安装前 `pip check` 已报告 `textworld 1.7.0 is not supported on this platform`；这是旧环境既有的问题，不能作为本次升级新增问题归因。

2026-09-07 安装后验证结果：

- PyTorch 2.10.0 / CUDA 12.8、torchvision 0.25.0、vLLM 0.19.1、Transformers 5.5.4、Accelerate 1.12.0；全部关键版本与依赖清单一致。完整安装版本记录在 `requirements/environment-20260907.freeze.txt`（包含历史包，是环境清单而非跨平台安装锁）。
- 两个模型的离线 AutoConfig、AutoProcessor、chat template 和视觉张量构造均通过。
- 在 Slurm 作业 212883 的 gpu-05、分配的单张 A100-SXM4-40GB 上，9B 的 Transformers BF16 图片问答通过，输出 `red`。
- 同一张卡上串行运行 9B vLLM BF16 图片问答通过，输出 `red`。vLLM 成功加载约 17.66 GiB 模型权重，首次引擎预热约 128 秒；这不是吞吐量 benchmark。引擎正常关闭，本次没有保留后台服务。
- 项目测试 25 passed；Ruff 检查、新增启动脚本 Bash 语法检查通过。
- cv2、NumPy、SciPy、scikit-learn、pandas、torchvision、timm、sentence-transformers、outlines 导入通过。旧 outlines 的 NumPy 冲突已通过升级 outlines 1.2.0 解决。
- `pip check` 仅剩安装前已经存在的 TextWorld 平台提示，未改动该非 Qwen 包。

限制与注意事项：27B 只完成配置/视觉预处理验证，当前单张 40GB allocation 不足以进行其完整 BF16 GPU 推理，需另行申请至少双 40GB 或单 80GB GPU 并验证实际负载。Transformers 的可选 `flash-linear-attention` / `causal-conv1d` 快速路径尚未安装，当前使用已通过验证的 PyTorch fallback；vLLM 使用其内置 Triton/FLA 与 FlashAttention 路径。vLLM 预热有上游 tensor-layout 提示、退出有 NCCL 清理提示，但最终推理通过且引擎正常退出。以上不代表所有真实空间工具或项目 PPO 链路已完成端到端验证。

已有 Python 进程或 Notebook 内核仍可能持有旧版模块；升级后请退出重启，再 `conda activate spatialcraft`。
