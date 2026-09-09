# Omni3D Qwen3.6-27B 正式任务

提交日期：2026-09-09（Asia/Dubai）。Slurm job **220679**，提交后状态见 `squeue`。

历史：原双卡任务 **218289 已取消**；单卡任务 **218301** 在启动验收因Qwen解码直接读取已卸载卷积权重而失败，正式实验未开始。现已完成局部代码修复，103项测试与真实27B单卡验收通过，提交 **220630**。原运行目录与冻结快照保留；定位与证据见 `docs/omni3d_27b_offload_fix.md`。

最新修复：**220630** 已通过GPU和27B推理验收，但在journal保存混合类型设备键时退出；修复序列化并补齐离线初始化检查后，107项测试通过，重新提交 **220679**。详情见 `docs/omni3d_27b_journal_fix.md`。

模型为现有官方 BF16 Qwen3.6-27B。申请 **1×A100 40GB + 128GB主机内存**、8 CPU、48小时。`--mem=128G` 指主机内存。BF16权重约52GiB，超过单卡容量，因此采用Transformers/Accelerate自动放置与CPU卸载：GPU0权重预算22GiB，CPU预算90GiB，留出空间工具和激活开销。保持BF16精度，不启用量化或磁盘卸载；CPU与GPU间权重传输会降低推理速度。启用 `PYTORCH_ALLOC_CONF=expandable_segments:True`。

算法配置与现有9B一致，仅更换backbone：thinking=false，单次输出/Skill生成4096 tokens，每题4条独立rollout，单pass，训练temperature=0.7/top_p=0.9，辅助与部署temperature=0。E/K容量100/20，每parent 6条轨迹、每轮最多2 parents、3候选、50/8步，PPO action-only，epsilon=0.2，strict positive improvement margin=0。

复用已准备的Omni3D实例级划分：251道训练题（1004条rollout）、250道冻结部署题。该配置独立初始化Experience和6个seed Skills。

- 配置：`configs/experiments/qwen36_27b_omni3d.yaml`
- 本地模型配置：`configs/models/qwen3.6-27b-local.yaml`；原vLLM配置保留。
- 运行目录：`/l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_journalfix_v3`
- 冻结代码：运行目录下 `code_snapshot/`
- 冻结提交/启动脚本：`launcher_code/`，文件哈希见其中的 `manifest.json`
- 提交与调度记录：`submission.json`
- 验证：离线preflight通过，107项测试通过，Ruff及shell语法检查通过。

GPU分配后先执行CUDA、27B单卡+CPU卸载加载、图像问答、seed重放、固定action评分和原生工具调用检查；全部通过后执行正式协议。GPU验收失败将停止，不进入正式任务。修复已在诊断220609通过真实27B单卡验收；正式任务仍会在分配的节点上重新验收。尚无完整benchmark准确率结果。

```bash
squeue -j 220679
squeue --start -j 220679
/home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python \
  /l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_journalfix_v3/launcher_code/status_robospatial.py \
  /l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_journalfix_v3 --dataset omni3d
```

日志位于 `launcher_logs/slurm-220679.log`，正式执行后另有 `*-execution.log`。最终结果为 `omni3d/results/deployment.json`。

若任务退出且未完成，先查明原因；确认无同run写入任务后，可用原 `launcher_code/omni3d_27b.sbatch` 续跑。它只执行冻结快照，不重新复制工作区代码。已提交阶段由journal复用，不能用新的参数覆盖旧run。
