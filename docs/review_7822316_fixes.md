# 7822316 三次复审处理说明

审计基线：`78223169447ffa16fac14133aded1ffae2c42e93`，分支 `thrid`。用户提供的 T01–T04 已逐项对照代码调用链，四项问题及修改方向均成立。本轮在此提交之上修复，不修改历史实验结果，也未执行 GPU/API benchmark。

## 问题判断与修复范围

| 项目 | 判断 | 已完成的修复 |
|---|---|---|
| T01：奇异相机内参导致积累/恢复卡住 | 相机契约遗漏可逆性与支持的矩阵结构，投影接受而反投影抛出 LinAlgError，继而被当成基础设施故障。属于非法参数分类错误。 | 投影、反投影统一有效内参契约，明确支持正对角、可逆 2×2 像素基的仿射内参及有限逆（允许非零 off-diagonal）；由矩阵输入导致的数值求逆失败转为 ToolSchemaError。Agent 通过参数错误观察修正下一动作，OOM/文件损坏等仍保留基础设施失败行为。 |
| T02：二维操作忽略操作数 frame | 2D distance/bbox 接受不同明示坐标系，返回的距离/关系缺乏空间含义；同坐标系公式本身没有问题。 | 2D/3D 共用操作数 frame 一致性检查；冲突时要求显式变换，同 frame 保持原计算结果。 |
| T03：选择 embedding 操作标签错误 | applicable 的临时 scope 在真正 embedding 前已退出，导致事件沿用 execution 标签。原始用量仍存在，影响的是按 operation 分项。 | 在真正 embedding 入口使用固定 skill.selection 标签，只改本次 ledger 副本；保留 phase/task/rollout 归属，并保持 generator scope 与后续请求身份稳定。 |
| T04：示例残留运行标识与本地路径映射 | ToolExecutor 注入的 invocation_id 与 resolved_artifact_uris 未被语义视图排除，相同语义示例随运行目录变化。 | 排除这两个运行审计字段，保留 artifact 依赖、frame 与尺度来源。原始 ToolResult、ArtifactStore/journal 继续保留完整执行记录。 |

没有需要反对的核心建议，但保持以下边界：不通过广泛捕获异常实现 T01；不把 T02 描述为距离公式错误；不把 T03 描述为用量全部丢失；T04 目前证明的是无关文本污染，未测量其检索或准确率影响。

补充处理了同类齐次矩阵问题：固定末行在明确绝对容差内的 SE(3)（1e-5）、内参和像素映射（1e-8，均 rtol=0），复制后规范化为精确 `[0,…,1]`，前向与求逆使用相同矩阵；超过容差则拒绝，不修改调用者数组。例如平移 `1e8` 且末行首项 `1e-8` 的输入原本通过 SE(3) 校验却在求逆处奇异，现在按所声明的 SE(3) 契约统一规范化。

`invert_matrix` 只将 `numpy.linalg.LinAlgError` 和非有限逆归类为 `ToolSchemaError`；保留合法的 skew/affine 像素标定，不凭“相机矩阵”名字额外收窄为仅对角焦距形式。

## 修改位置

- T01/T02：`src/spatialcraft/tools/spatial_arrays.py`、`src/spatialcraft/tools/builtin/geometry.py`，Geometry 工具版本更新为 2.1.1；[矩阵、坐标系与 Runtime 恢复测试](../tests/test_geometry_input_recovery_t01_t02.py)。
- T03：`src/spatialcraft/experiments/usage.py`、`src/spatialcraft/experiments/runtime_v2.py`；[训练、部署和恢复测试](../tests/test_selection_embedding_scope_v2.py)。选择算法及 LLM 判断流程保持原定义。
- T04：`src/spatialcraft/knowledge/evidence.py`；[真实跨目录工具链测试](../tests/test_real_tool_demonstration_evidence.py)。无需改动 ToolExecutor 审计写入或删除整份 metadata。

## 验证与恢复边界

本轮使用实际 GeometryTool、ToolExecutor、ArtifactStore 和正式 Runtime 的 CPU 回归；模型与 embedding 为脚本化提供者。T01 的 Runtime 测试显式关闭 Experience/Skill 分支以隔离工具纠错，每题仍执行四条 rollout；T03 测试包含默认记忆流程的积累和冻结部署。验收分别覆盖：非法相机矩阵后修正并完成、中断恢复；二维相同/冲突 frame；积累与部署的选择成本归属和恢复请求一致；跨目录同语义工具链示例一致且原审计可追溯。最终完整回归 **433 passed，15 warnings，77.01 秒**，无失败或跳过；比审计提交新增 20 项测试（T01/T02 15 项、T03 3 项、T04 2 项）。8 个修改过的 Python 文件 Ruff check / format --check 通过，`git diff --check` 通过。警告来自 torch NVML 初始化及 matplotlib/pyparsing 弃用接口。测试前后源码摘要一致。

[可随源码审阅的 CPU 结果与文件 SHA256](validation/review_7822316_cpu.json)；[本地完整测试输出](../artifacts/v2_validation/review_7822316_cpu/pytest.txt)。本轮修改尚未提交；上轮 413 项结果保留为旧版本记录。复跑命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/xiwei.liu/miniconda3/envs/spatialcraft/bin/python -m pytest -q
```


修复不重写旧 ledger；历史 T03 记录不会自动得到新标签。示例清理也不自动重建已冻结的旧 RAG 库。新回归验证本轮代码的正常与恢复路径；已有实验仍受代码/配置身份绑定，不能据此绕过 journal 的版本检查而混合新旧实现。
