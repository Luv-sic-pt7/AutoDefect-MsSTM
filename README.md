# 融合深度语义与主题模型的汽车质量缺陷智能感知与决策支持系统 (MsSTM & CRWTM)

[cite_start]**项目编号**: X202510533649 [cite: 182]  
[cite_start]**项目负责人**: 周玟帅 [cite: 183]  
[cite_start]**指导教师**: 施文 [cite: 188]

## 项目简介
本项目旨在解决海量非结构化汽车质量投诉数据中的语义理解深度不足与风险预测滞后问题。通过构建 **基于元数据的监督型段落主题模型 (MsSTM)** 和 **投诉召回加权主题模型 (CRWTM)**，实现从“短文本+元数据”中精准提取缺陷主题，并预测召回风险。

## 核心特性
- [cite_start]**深度语义融合**: 集成 BERT-base-chinese 预训练模型，提取 768 维语义向量 [cite: 242]。
- **结构化建模**: 构建 "Brand-Model-Year (MMY)" 三级标注体系。
- [cite_start]**监督学习**: 引入泊松-狄利克雷过程 (PDP) 与 Logistic Regression 联合建模 [cite: 237]。
- [cite_start]**高效推断**: 实现分层 CRP-Gibbs 采样与 Stochastic EM 算法 [cite: 244]。

## 环境依赖
- Python 3.8+
- [cite_start]TensorFlow 2.13 [cite: 245]
- Transformers 4.30.0
- Jieba

## 快速开始
```bash
python main.py --mode train --topics 20 --alpha 0.1 --beta 0.01