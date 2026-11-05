# 交付验证记录

验证日期：2026-11-04。以下为本地工程验证，不是完整 ModelNet40 性能评估。

## 实际环境

| 项目 | 版本 |
|---|---|
| 系统 | Windows |
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| NumPy | 2.4.4 |
| scikit-learn | 1.9.0 |
| h5py | 3.16.0 |
| tqdm | 4.69.0 |
| Matplotlib | 3.11.1 |

`requirements.txt` 给出依赖范围，以上是实际验证组合，不代表范围内每个版本都已测试。

## 回归检查

```powershell
python tests/test_portability.py
```

6 项测试覆盖：

1. 含空格的数据目录、环境变量路径、显式参数覆盖，以及两个 Dataset 的输出形状。
2. 不支持的 split、点数不足等输入错误。
3. DGCNN 图特征和 DGCNN / PointNet 基线的单样本 CPU 推理。
4. PointNet 分类损失中有/无特征变换矩阵两种分支及反向传播。
5. PointNet++ MSG 的 CPU 前向传播。
6. 六个命令行入口的 `--help` 与数据路径参数。

## 训练与检查点冒烟测试

```powershell
python tests/smoke_training.py
```

脚本在临时目录生成 40 类合成文本点云，每类 1 个训练样本和 1 个测试样本、每个样本 32 个点。所有生成数据和权重在完成后删除，不使用或覆盖个人实验目录。

- PointNet：CPU 训练 1 epoch，写入 `best_model.pth`，从独立评估进程读取该检查点。
- DGCNN：CPU 训练 1 epoch，保存并重新读取 `.t7`；测试 batch size 为 13，使最后一批只有 1 个样本。
- 将新权重转换为带 `module.` 前缀的旧式 DataParallel 格式，再验证评估加载兼容性。
- 测试机器具有可用 GPU，但通过参数强制 CPU，覆盖“有显卡仍要求 CPU”这一场景。

合成数据只用于确认前向、反向、保存、加载和参数传递可运行，准确率没有模型评估意义。

## 真实数据采样演示

设置 `MODELNET40_ROOT` 后，在主仓库运行：

```powershell
python visualization_pointnet2_sampling.py --sample_index 0 --seed 0 --output docs/assets/sampling_demo.png
```

真实样本 `airplane_0001`：10,000 个输入点，FPS 保留 1,024 个点、选择 128 个中心。半径 0.1、邻域上限 16，选中中心有 11 个半径内点，返回 16 个索引、11 个不同索引。输出图已人工查看，统计记录在同目录 JSON 中。

## 未验证范围

- 未用整理后的代码完成完整 ModelNet40 训练或性能复测。
- 未验证 GPU / 多 GPU 训练、Linux 执行和多进程 DataLoader。
- 未验证上游 TensorFlow、部件分割、语义分割与原始 Open3D/编译可视化工具。
- 未对历史 checkpoint 做精度复核，也未执行历史重写。
- 本次验证范围为本地工作区，未验证远程仓库的公开访问与干净克隆。
