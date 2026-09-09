# Qwen3.6-27B 单卡 CPU 卸载修复（2026-09-09）

任务218301在启动验收退出，尚未创建Omni3D实验journal，也未调用付费embedding。CUDA采样断言报告概率包含NaN/Inf/负值。

## 定位

本地Transformers 5.5.4的Qwen3_5GatedDeltaNet在prefill调用`conv1d(...)`，缓存解码却直接读取`conv1d.weight`。Accelerate在子模块调用前加载权重、调用后卸载；直接读权重绕过加载钩子，读取到meta占位张量。

真实Qwen3.6-27B诊断在gpu-08、单张A100 40GB上复现：`model.language_model.layers.24.linear_attn.conv1d.weight is meta during cached decode`。诊断先在执行有问题的CUDA卷积之前拦截缺失权重，再在同一个模型上应用修复进行验收。

## 修改

`src/spatialcraft/models/providers/offload.py`在模型加载后，将使用独立卸载钩子的Qwen卷积权重从CPU权重映射还原到其执行设备，并移除该卷积的卸载钩子。大投影层继续按原配置卸载，更新hf_device_map供验收记录。修复可重复调用，未卸载的9B模型不受影响。

27B当前配置共有30个受影响的卷积层，增加2,457,600字节（约2.34MiB）GPU权重。模型仍为BF16，不清洗/替换NaN，不改变采样、缓存、评分或实验超参数。

## 验证与记录

- 回归测试使用真实Qwen模块及Accelerate CPU卸载，覆盖故障复现、prefill、连续解码、重复请求和无缓存评分；与常驻权重结果逐元素完全一致。
- 完整测试103项通过（30.35秒）。登录节点的NVML告警不影响CPU测试，GPU能力由独立验收检查。
- 真实27B诊断任务220609，记录目录：`/l/users/xiwei.liu/spatialcraftLog/validation/omni27b_offload_20260909_healthy`。
- 先前诊断220608在gpu-51因CUDA驱动初始化失败退出，未运行模型；重新采用原先的故障节点排除列表。
- 27B图像问答、种子重放、固定action评分、原生工具调用和跨请求缓存解码验收：全部通过。图像回答Red，多步重放回答Red.；峰值allocated显存23,651,193,856字节（约22.03GiB）。provider.json与extended_decode.json均为passed。

旧任务及冻结快照完整保留。修复后的独立正式运行目录为`/l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_offloadfix_v2`，正式任务 **220630** 已提交，状态以Slurm为准；监控入口见`docs/omni3d_27b_run.md`。

相关上游说明：Accelerate官方文档解释了直接读取子模块权重需要额外处理的原因（https://huggingface.co/docs/accelerate/package_reference/big_modeling）；新版Transformers Qwen实现也在forward上对conv1d使用force_accelerate_hooks（https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py）。本项目保留当前依赖版本，采用局部兼容修复。
