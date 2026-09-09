# Omni3D journal 初始化修复（2026-09-09）

任务 **220630** 在gpu-06完成CUDA与27B图像问答、seed重放、固定action评分和原生工具调用验收后，于正式实验的RunJournal初始化退出。异常为 `TypeError: '<' not supported between instances of 'str' and 'int'`。尚无实验journal、训练轨迹或付费embedding调用。

## 原因与修复

模型的max_memory合法配置为 `{0: "22GiB", "cpu": "90GiB"}`。GPU加载需要整数设备编号；RunJournal保存模型配置时，在键尚未转换为JSON字符串前执行sort_keys，Python无法比较整数与字符串。

修改 `src/spatialcraft/experiments/journal.py`：先复制成JSON表示，再按既有canonical规则进行哈希。模型内存中的整数设备键不变；已有规范记录的校验值与断点绑定语义不变。对转换后重名的键（例如0与"0"）明确拒绝，仍拒绝NaN/Inf。

新增 `check_runtime_initialization`，在默认离线preflight中实际创建ExperimentRuntime、数据集pipeline和journal，提交一个临时检查阶段，再重新打开和校验。该检查不会加载模型或创建API客户端，不写正式运行目录。GPU启动验收也增加真实pipeline初始化，覆盖本次遗漏。

## 验证

- 完整测试 **107项通过，29.21秒**；Ruff与shell语法检查通过。
- 真实Omni3D配置离线检查通过：journal创建、提交、重开均正常，251题训练/1004条rollout/250题部署。
- 回归覆盖9B/27B真实runtime初始化、混合键记录与断点复用、参数变化拒绝、键冲突和非有限值拒绝；测试强制禁止调用模型加载与embedding。
- Qwen推理provider与卷积卸载修复源码逐字节保持与220630成功GPU验收版本一致，新作业分配后仍会重跑GPU验收。

## 新任务

**220679** 已提交，Qwen3.6-27B、thinking=false、4096 tokens、每题4 rollouts、单pass、单张A100 40GB、128GB主机内存。运行目录：`/l/users/xiwei.liu/spatialcraftLog/runs/omni3d_qwen36_27b_instruct4096_single_a100_journalfix_v3`。

旧run、日志及冻结快照全部保留；旧任务尚无训练结果，本次从第一题开始。实时状态以Slurm和新运行目录为准。
