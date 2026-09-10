# GPT-5.4 三轮纯视觉 baseline

2026-09-10 已提交：**226631（run1）**、**226632（run2）**、**226633（run3）**；汇总 **226634** 依赖三轮全部成功，`afterok:226631:226632:226633`。实际运行状态以 Slurm 和 progress 文件为准。

## 配置与范围

使用 **gpt-5.4**，reasoning=none、temperature=0、每次最多4096 tokens。每题一次独立作答、单pass，仅输入原始图像、题目和选项；无工具、Experience、Skill、embedding或PPO。全部视图按原顺序输入，detail=high。Responses API不支持生成seed，run1/run2/run3只是重复评测编号。

RoboSpatial 175、ERQA 200、Omni3D 250、SAT 300，每轮925题，三轮2775次正式作答。数据与已完成的gpt-5.4-mini baseline逐字节相同，沿用固定deployment划分和SAT circular 300，原图和评分器不变。原有mini结果不纳入本次GPT-5.4汇总。

每个评测作业4 CPU、16GB主机内存、24小时；汇总1 CPU、2GB、30分钟，均使用cscc-cpu-p，无GPU。每轮四个数据集各一个线程，生成及评分逐题落盘；模型调用断点复用，网络/服务错误经SDK重试后仍失败则停止对应数据集，其他数据集继续。输出截断/拒答/无答案按单回答baseline计零分，无额外recovery。

## 查看结果

运行组目录：`/l/users/xiwei.liu/spatialcraftLog/runs/gpt54_direct_baseline_20260910_three_runs_v1`。

- `run{1,2,3}/{dataset}/progress.json`：实时进度和当前部分准确率。
- `run{1,2,3}/{dataset}/results/deployment.json`：该轮该数据集完整准确率和分类统计。
- `run{1,2,3}/{dataset}/results/predictions.jsonl`：逐题答案、reward、评分、实际模型版本和token用量。
- `run{1,2,3}/launcher_logs/slurm-{job_id}.log`：三轮执行日志。
- `results/accuracy_mean_std.json`：三轮分数、mean、样本std、variance、分类统计和来源校验值。
- `results/accuracy_mean_std.csv`、`results/accuracy_mean_std.md`：三轮accuracy与mean±std表格。
- `launcher_logs/aggregate-226634.log`：汇总日志。
- `group.json`、`submission.json`、`validation/`：参数、作业依赖和验收记录。

主要std使用样本标准差 **ddof=1**，另存population std；JSON中mean/std为accuracy比例，variance为比例平方，表格用百分比且std为百分点。汇总会拒绝不完整结果、模型版本/协议/题目集合不一致、不同轮复用response IDs或逐题计分与accuracy不一致的结果。只有三轮全部完成才写出最终汇总。

```bash
squeue -j 226631,226632,226633,226634
```

## 代码与验证

执行入口 `src/spatialcraft/experiments/run_api_baseline.py` 新增 `--model-config`，本次指定 `configs/models/gpt-5.4-baseline.yaml`。原gpt-5.4通用角色配置仍保留其原参数。`scripts/inference/gpt54_baseline.sbatch` 为单轮入口，`gpt54_baseline_aggregate.sbatch` 为三轮汇总入口；实际运行均使用各目录下冻结代码。

汇总器模型标题从实际journal绑定读取，不再固定写mini；返回模型校验只接受请求的alias或该alias的日期版本，避免gpt-5.4-mini/pro/nano前缀误判为gpt-5.4。基于两模型的断点/评分/汇总及模型替换检查，共 **33项测试通过（0.74秒）**；Ruff和shell语法通过；三份冻结代码离线preflight通过；全部输入与mini的原始925题逐字节一致。

真实API的16张合成图验收通过，返回 `gpt-5.4-2026-03-05`、答案red、status=completed、reasoning_tokens=0。该合成验收不计入正式2775题，也不用于调整提示词。

[OpenAI GPT-5.4 参数文档](https://developers.openai.com/api/docs/models/gpt-5.4)。
