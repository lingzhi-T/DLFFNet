# DLFFNet: Dynamical Local Feature Fusion Network for AVC Recognition

本项目实现并复现了 DLFFNet，一种用于自动识别主动脉瓣钙化（Aortic Valve Calcification, AVC）的动态图像深度学习模型。该模型结合多尺度特征、局部区域融合、掩膜调节机制与金字塔时序建模模块，专为超声心动图序列设计。

---

## 📁 项目结构说明

- `train_new-store-path_4fold_gccs-fpn-branch-local-feature-fusion-changemask_3_classification_20240116.py`  
  主训练脚本。实现4折交叉验证下的训练流程，包含数据加载、模型构建、loss计算、ROC绘制、结果保存等。

- `test_gccs_3_classification_0530.py`  
  模型评估脚本。用于加载训练好的模型并进行测试评估，输出准确率、ROC曲线等。

- `functions.py`  
  训练版数据加载模块，主要提供图像与掩膜加载逻辑，仅返回 loss 和 output，适合大规模训练。

- `functionsbranch.py`  
  论文版数据加载模块，包含 STN 局部裁剪、掩膜调节、注意力可视化等中间变量，主要用于分析与可视化展示。

- `our_model.py`  
  模型定义文件。包含多种模块封装，包括：
  - 多级卷积块 `Conv2dBlock`
  - 残差模块 `ResBlock`
  - 通道注意力与空间注意力（CBAM）
  - 金字塔结构、局部增强等核心模块

- `0001.png`  
  模型结构图（示意图），对应 DLFFNet 中的 Fig.1。

---

## 🔧 模型结构简介

DLFFNet 核心结构包括：

1. **数据预处理模块**
   - 使用 Faster R-CNN 对主动脉瓣区域进行自动定位；
   - 使用 U-Net 分割高回声区域生成掩膜；
   - 掩膜用于后续区域增强与特征调节。

2. **双分支中融合主干网络**
   - 图像分支：使用 ResNet18 提取多层语义特征；
   - 掩膜分支：轻量结构，嵌入 MTM 与 LFFM 进行交叉调节；
   - MTM（Mask Tuning Module）用于细化掩膜；
   - LFFM（Local Feature Fusion Module）用于局部特征选择与增强。

3. **金字塔时序融合模块（PBTBFFM）**
   - 多层特征融合后，接入 LSTM 进行时间建模；
   - 最终输出分类结果（AVC风险等级）。

---

## ⚙️ 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动训练
```bash
python train_new-store-path_4fold_gccs-fpn-branch-local-feature-fusion-changemask_3_classification_20240116.py
```

### 启动测试
```bash
python test_gccs_3_classification_0530.py
```

---

## 📊 实验数据说明

- 图像来源：第一医院收集的超声心动图，231 名患者，包含短轴视图；
- 每个序列包含多个切片图像及对应掩膜；
- 标签包括主动脉瓣钙化风险等级（3类）与生存时间等信息。

---

## 🧠 主要创新点

- 提出局部特征融合模块 LFFM，实现基于掩膜指导的区域增强；
- 掩膜调节模块 MTM 可缓解分割误差影响；
- 多尺度 + 金字塔结构建模，配合 LSTM 捕捉时间动态；
- 支持多模态输入与结构可视化分析。

---

## 📎 补充说明

若进行论文撰写或可视化分析，推荐使用 `functionsbranch.py` 与可视化模块辅助生成中间结果（如掩膜效果图、注意力热图等）。

