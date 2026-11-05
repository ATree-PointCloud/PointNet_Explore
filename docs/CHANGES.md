# 我的源码改动记录

## 整理前已经存在的实验改动

PointNet 子仓库相对本地记录的上游提交 `eb64fe0b4c24055559cea26299cb485dcb43d8dd`：

- `models/pointnet_cls.py`：我关闭了 feature transform；输入空间的 STN3d 仍然保留。损失在没有特征变换矩阵时仅计算分类项。
- `models/pointnet2_cls_msg.py`：我把第一层 MSG 半径从 `[0.1, 0.2, 0.4]` 扩大到 `[0.2, 0.4, 0.8]`，保留中心数 512、邻域上限 `[16, 32, 128]` 和 MLP 配置。
- `models/pointnet2_utils.py`：我增加 `debug` 与 `debug_printed`，首次前向传播输出每个尺度的半径、分组形状和第一个中心的去重邻居数。该统计只观察一个中心，不能代替整层邻域分布。
- `models/pointnet2_part_seg_msg.py` / `pointnet2_sem_seg_msg.py`：我也开启了第一层调试打印，但未在本次交付中验证分割任务。
- 分类入口曾使用我的 Windows 绝对数据路径，现已改为参数。

DGCNN 子仓库基于本地记录的上游提交 `f765b469a67730658ba554e97dc11723a7bab628`：

- `pytorch/data.py`：我把 ModelNet40 分类数据改为读取 normal-resampled 文本文件，使用 split 列表和类别名称映射；保留 XYZ，截取前 N 个点，训练时缩放、平移并打乱顺序。保留上游 HDF5 辅助函数作为历史实现，但当前 Dataset 不调用它们。
- `pytorch/model.py`：我为邻居索引展开和图特征维度补充了注释，没有替换 EdgeConv 主干。
- 主仓库的 `visualization_pointnet2_sampling.py` 是我用于理解 FPS 和 Ball Query 的独立实验入口。

## 本次交付整理

- 分类训练、评估、数据检查和演示均支持 `--data_root` / `MODELNET40_ROOT`；独立子仓库不依赖主仓库的路径辅助模块。
- DataLoader 默认 `--workers 0`，可自行调高；缺失数据时提供配置提示。修复 PointNet 数据检查入口缺少 `args` 的问题。
- DGCNN 将 shell `cp` 改为 `shutil.copy2`，使用输入张量所在设备创建图索引。
- DGCNN 评估标签使用 `view(-1)`，PointNet 基线池化使用 `squeeze(-1)`，避免最后一批只有一个样本时丢失 batch 维度。
- DGCNN 布尔参数可正确解析 false，评估按 `--model` 构建模型并映射到当前设备。
- PointNet 分类评估的投票张量跟随输入设备；检查点按所选设备加载。仅加载自己生成或信任的检查点。
- PointNet 特征正则项改用 `trans_feat is not None` 判断，避免 Tensor 布尔判断异常。
- 两个训练入口的学习率调度更新移至每个 epoch 的优化步骤之后。
- 采样演示增加参数、固定种子、归一化、无窗口 PNG 与 JSON 导出。修正原脚本打印“512 个中心”但实际取 128 个的不一致，删除未使用的 Open3D 依赖。
- 补充测试、环境依赖与来源说明。

## 保留的边界

没有将本次工程整理描述为模型创新，也没有宣称新的精度提升。没有改写已有 Git 历史；历史中的权重、日志和旧路径仍然存在。原始 TensorFlow 与分割入口保留上游行为，不属于此次可移植性承诺范围。现有 PointNet 训练恢复只恢复模型和 epoch，不能视为优化器、调度器状态完全连续的断点续训。
