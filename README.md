# NeurIPS – Open Polymer Prediction 2025 (Kaggle)

本仓库为 **NeurIPS Open Polymer Prediction 2025** 竞赛的解决方案。任务为对高分子聚合物（CRU SMILES 表示）同时预测五项关键物性：

- `Tg`
- `FFV`
- `Tc`
- `Density`
- `Rg`

目的在于利用机器学习加速新型可持续材料的筛选与设计。

---

## 🎯 方法概览

本方案基于 **图神经网络（GNN）**，将每条聚合物 CRU 表示为图结构，并融合全局分子描述符进行多任务预测。

### 图表示设计

| 级别 | 特征内容 |
|---|---|
| **节点特征** | 原子号、价态、电荷、杂化、芳香性、是否在环、氢数、质量等 |
| **边特征** | 键型（单/双/三/芳香）、共轭性、环性、立体信息 |
| **全局特征** | RDKit Descriptors、EState 汇总、Gasteiger 电荷统计、SMARTS 片段计数、连接点距离等 |

### 模型结构：CRU-GNN

- 多层 **GINEConv** 消息传递
- 池化 = **全局均值池化 + 注意力池化**
- 全局图嵌入与分子描述符融合
- **多任务 wMAE 损失**（自动平衡不同物理量的量纲差异）
- **EdgeDrop + GraphNorm + EMA** 提升泛化稳定性

---

## 📊 比赛表现

| 指标 | 成绩 |
|---|---|
| Public Leaderboard | **0.08699** |
| Private Leaderboard | **0.06300** |
| 排名 | Top 区间（具体名次随最终封榜情况） |

模型具有良好的稳定性与可复现性。


## 📁 项目结构


---

## 🚀 快速使用（在 Kaggle Notebook 中）

将 `final_submit.py` 上传后直接运行：

```bash
python final_submit.py

脚本将自动完成：

数据加载

5-Fold 训练

最优模型集成

生成提交文件 submission.csv

模型权重将保存在：
/kaggle/working/checkpoints/cru_gnn_best_fold*.pt
🔍 推理与生成提交
单模型预测

from final_submit import predict_and_make_submission

predict_and_make_submission(
    ckpt_path='checkpoints/cru_gnn_best_fold0.pt',
    test_csv='test.csv',
    sample_sub_csv='sample_submission.csv',
    out_path='submission.csv'
)
多模型集成

from final_submit import predict_ensemble_and_make_submission

predict_ensemble_and_make_submission(
    ckpt_paths=['cru_gnn_best_fold0.pt', 'cru_gnn_best_fold1.pt', ...],
    test_csv='test.csv',
    sample_sub_csv='sample_submission.csv',
    out_path='submission.csv'
)
✨ 关键创新点总结

Star-aware ECFP 表示方式：保留聚合物连接点的结构语义

图结构表征 + 分子描述符融合：兼顾局部与全局信息

基于化学物理意义的 wMAE 做量纲与数据平衡

EdgeDrop + EMA 系列泛化增强策略，使模型在 leaderboard 上表现稳定

🙌 致谢

感谢 NeurIPS 组委会与 Kaggle 社区为开放材料科学研究提供高质量平台。
