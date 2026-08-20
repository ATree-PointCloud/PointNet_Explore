import os
import numpy as np
import torch
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import farthest_point_sample, index_points, query_ball_point

import open3d as o3d


DATA_PATH = 'C:/Users/naihui.zhang/data/modelnet40_normal_resampled/'
SAMPLE_INDEX = 0


def main():
    cat = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'modelnet40_shape_names.txt'))]
    classes = dict(zip(cat, range(len(cat))))
    print("分类标签:", classes)

    train_shape_ids = [line.rstrip() for line in open(os.path.join(DATA_PATH, 'modelnet40_train.txt'))]
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in train_shape_ids]
    datapath = [(shape_names[i], os.path.join(DATA_PATH, shape_names[i], train_shape_ids[i]) + '.txt') for i in range(len(train_shape_ids))]

    fn = datapath[SAMPLE_INDEX]
    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
    point_cls = fn[0]
    print("示例数据原始维度:", point_set.shape)
    print(f"示例数据标签: {point_cls}({classes[point_cls]})")

    print("====================处理点云数据，仅选用1024个point=======================")
    point_set = torch.from_numpy(point_set).unsqueeze(0)  # [B=1, N, C]
    xyz = point_set[:, :, :3]  # [B=1, N, 3] FPS只使用坐标
    fps_idx = farthest_point_sample(xyz, 1024)  # [B=1, 1024] 采样点索引
    point_set = index_points(point_set, fps_idx)  # [B=1, 1024, C] 按索引取出采样点
    print("示例数据处理后维度:", point_set.shape)

    print("=====================从1024个点中采样512个centroid，并得到中心坐标========================")
    xyz = point_set[:, :, :3]
    centroid_idx = farthest_point_sample(xyz, 128)
    centroid_point = index_points(point_set, centroid_idx)

    print(f"centroid point维度为: {centroid_point.shape}")
    print(f"原始point_set维度为: {point_set.shape}")

    point_set_np = point_set[0].detach().cpu().numpy()
    centroid_point_np = centroid_point[0].detach().cpu().numpy()
    complement_mask = ~np.isin(np.arange(point_set_np.shape[0]), centroid_idx[0].cpu().numpy())
    complement_point_np = point_set_np[complement_mask]
    print(f"centroid point numpy维度为: {centroid_point_np.shape}")
    print(f"补集点numpy维度为: {complement_point_np.shape}")

    # point_set_pcd = o3d.geometry.PointCloud()
    # point_set_pcd.points = o3d.utility.Vector3dVector(point_set_np[:, :3])
    # point_set_pcd.normals = o3d.utility.Vector3dVector(point_set_np[:, 3:6])

    selected_centroid = centroid_point[:, 0:1, :3]
    print("选中centroid维度为:", selected_centroid.shape)

    grouped_idx = query_ball_point(0.1, 16, xyz, selected_centroid)
    print("Ball Query 返回grouped_idx维度:", grouped_idx.shape)

    grouped_points = index_points(point_set, grouped_idx)
    grouped_points_np = grouped_points[0].detach().cpu().numpy()  # (1, 16, 6)
    grouped_points_np = grouped_points_np.reshape(-1, 6)  # 摊平为 (16, 6)，Open3D需要(N, 3)
    print("grouped_points_np维度:", grouped_points_np.shape)

    # colors = np.full((point_set_np.shape[0], 3), 0.5)  # 全部初始化为背景灰色
    # group_idx_np = grouped_idx[0].reshape(-1).cpu().numpy()  # (16,) group点在point_set中的索引
    # colors[group_idx_np] = [0, 1, 0]  # group：绿色
    # colors[centroid_idx[0, 0].item()] = [1, 0, 0]  # 选中centroid：红色（若在group内则覆盖绿色）
    #
    # combined_pcd = o3d.geometry.PointCloud()
    # combined_pcd.points = o3d.utility.Vector3dVector(point_set_np[:, :3])
    # combined_pcd.normals = o3d.utility.Vector3dVector(point_set_np[:, 3:6])
    # combined_pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([combined_pcd])  # type: ignore[arg-type]

    center = selected_centroid[0, 0]
    dist = torch.sqrt(torch.sum((xyz[0] - center)**2, dim=1))
    valid_mask = dist <= 0.1

    print("radius内真实点数:", valid_mask.sum().item())
    print("Ball Query输出点数:", grouped_idx.numel())
    print("Ball Query输出unique点数:", torch.unique(grouped_idx).numel())


if __name__ == '__main__':
    main()