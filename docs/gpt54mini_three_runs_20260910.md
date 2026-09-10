# GPT-5.4-mini 三轮 baseline 汇总

2026-09-10：首轮 **226209** 已完成，另外两轮 **226421（rep43）**、**226422（rep44）** 已提交；汇总作业 **226423** 依赖两轮成功（`afterok:226421:226422`），自动统计首轮加两次复跑的结果。实时状态以 Slurm 为准。

## Seed 与实验含义

当前 OpenAI Responses API 和本地 SDK/provider 不支持生成 seed。用户要求的 43、44 仅记录为 `repetition_label`，`api_seed=null`、`seed_applied_to_model=false`。首轮也未给模型设置 seed，原 seed42 是数据划分的种子。三轮是保持相同条件的独立 API 重复评测，不声称实现了可控模型种子的实验。

三轮均保持 gpt-5.4-mini、reasoning=none、temperature=0、4096-token 上限、每题一次作答、不使用工具/Experience/Skill。新任务实际重新调用 API，不复用首轮模型响应。模型可能在相同参数下产生相同结果，std=0 也属有效结果。

测试集合、题序、图片、提示词、评分器和源码与首轮一致；全部代码及数据逐字节验证。RoboSpatial 175、ERQA 200、Omni3D 250、SAT 300，每轮925题，两次新增1850次正式评测。复跑各申请4 CPU、16GB、24小时；汇总1 CPU、2GB、30分钟；没有GPU资源请求。

## 结果与进度

首轮：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_direct_baseline_20260910_v1`。

本次组目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54mini_direct_baseline_20260910_repeats_v1`。

- `rep43/{dataset}/progress.json`、`rep44/{dataset}/progress.json`：实时进度。
- `rep43/{dataset}/results/deployment.json`、`rep44/{dataset}/results/deployment.json`：单轮总体及分类accuracy。
- `rep43/launcher_logs/slurm-226421.log`、`rep44/launcher_logs/slurm-226422.log`：实时执行日志。
- `results/accuracy_mean_std.json`：三轮原始accuracy、mean、std、variance及分类统计。
- `results/accuracy_mean_std.csv`、`results/accuracy_mean_std.md`：便于查看/引用的三轮分数与mean±std表。
- `launcher_logs/aggregate-226423.log`：自动汇总日志。
- `group.json`、`submission.json`、`rep*/repetition.json`：运行映射、作业依赖、真实seed语义。

`mean`、`std` 以accuracy比例[0,1]记录，`variance` 为比例的平方；主要std是**样本标准差，ddof=1**，并附 `std_population`（ddof=0）。表格使用百分比，std以百分点表示。仅计算同一数据集三轮准确率之间的波动，不将925题混在一起计算一个std，也不将2750余条二元reward的离散程度当作三轮std。

首轮分数：RoboSpatial 92/175（52.57%）；ERQA 69/200（34.50%）；Omni3D 56/250（22.40%）；SAT 173/300（57.67%）。另两轮未结束前不生成最终均值和标准差。

## 汇总校验与恢复

汇总器 `scripts/inference/aggregate_api_baselines.py` 检查三份目录互异、所有数据集完整、任务ID/顺序相同、journal协议完全一致、实际返回模型版本一致、模型response IDs不跨轮复用，且逐题reward与最终accuracy一致。任何检查失败都拒绝输出最终汇总。

16项针对baseline和汇总的回归通过（0.61秒），验证均值/样本std/variance及不完整、参数变更、重复响应、题目变更、模型版本变化等拒绝路径；Ruff、shell语法、两份冻结快照离线检查通过。首轮结果校验值保存在 `validation/first_result_hashes.json`，未改写首轮结果。

若某次API任务失败，已有journal可使用该复跑的冻结脚本继续；汇总作业不会将部分结果当作完整实验。续跑使用新job ID时应重新安排相应afterok汇总依赖。

API参数依据：[Responses create](https://developers.openai.com/api/reference/python/resources/responses/methods/create)。
