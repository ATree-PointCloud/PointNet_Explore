"""Inspect FPS and ball-query neighborhoods; optionally export a static figure."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import farthest_point_sample, index_points, query_ball_point


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, default=os.environ.get('MODELNET40_ROOT', str(Path(__file__).resolve().parent / 'data/modelnet40_normal_resampled')))
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--centroids', type=int, default=128)
    parser.add_argument('--radius', type=float, default=0.1)
    parser.add_argument('--neighbors', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Export a PNG without opening a window')
    return parser.parse_args()


def main(args):
    root = args.data_root.expanduser()
    split = root / 'modelnet40_train.txt'
    if not split.is_file():
        raise FileNotFoundError(f'Missing {split}. Set --data_root or MODELNET40_ROOT.')
    shape_ids = split.read_text(encoding='utf-8').splitlines()
    if not 0 <= args.sample_index < len(shape_ids):
        raise ValueError('sample_index is outside the training split')
    shape_id = shape_ids[args.sample_index]
    category = shape_id.rsplit('_', 1)[0]
    points = np.loadtxt(root / category / (shape_id + '.txt'), delimiter=',', dtype=np.float32)
    xyz_np = np.atleast_2d(points)[:, :3].copy()
    if not (1 <= args.centroids <= args.num_points <= len(xyz_np)):
        raise ValueError('require 1 <= centroids <= num_points <= input point count')
    if not (1 <= args.neighbors <= args.num_points) or args.radius <= 0:
        raise ValueError('require radius > 0 and 1 <= neighbors <= num_points')
    xyz_np -= xyz_np.mean(axis=0)
    scale = np.linalg.norm(xyz_np, axis=1).max()
    if scale == 0:
        raise ValueError('Cannot normalize a cloud with zero spatial extent')
    xyz_np /= scale
    torch.manual_seed(args.seed)
    xyz = torch.from_numpy(xyz_np).unsqueeze(0)
    sampled = index_points(xyz, farthest_point_sample(xyz, args.num_points))
    centroid_indices = farthest_point_sample(sampled, args.centroids)
    centroids = index_points(sampled, centroid_indices)
    center = centroids[:, :1, :]
    group_indices = query_ball_point(args.radius, args.neighbors, sampled, center)
    distances = torch.linalg.vector_norm(sampled[0] - center[0, 0], dim=1)
    report = dict(shape_id=shape_id, seed=args.seed, input_points=len(xyz_np), sampled_points=args.num_points,
                  centroids=args.centroids, radius=args.radius, requested_neighbors=args.neighbors,
                  points_in_radius=int((distances <= args.radius).sum()), returned_neighbors=group_indices.numel(),
                  unique_returned_neighbors=torch.unique(group_indices).numel())
    print(json.dumps(report, indent=2))
    if args.output:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cloud = sampled[0].numpy()
        centers = centroids[0].numpy()
        neighbors = cloud[np.unique(group_indices.numpy())]
        fig = plt.figure(figsize=(13, 4.5), layout='constrained')
        titles = ['Input: centered / unit sphere', f'FPS: {args.num_points} points, {args.centroids} centroids',
                  f'Ball query: radius {args.radius:g}, {len(neighbors)} unique neighbors']
        for i, title in enumerate(titles):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d', computed_zorder=False)
            background = xyz_np if i == 0 else cloud
            ax.scatter(*background.T, s=1, color='#a6b7ca', alpha=0.45)
            if i == 1:
                ax.scatter(*centers.T, s=8, color='#1265ad')
            if i == 2:
                ax.scatter(*neighbors.T, s=18, color='#20a387')
                ax.scatter(*center[0].numpy().T, s=65, marker='x', linewidths=2, color='#d55e00', depthshade=False)
            ax.set(title=title, xlabel='x', ylabel='y', zlabel='z')
            ax.set_box_aspect((1, 1, 1))
            ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[-1, 0, 1])
            ax.view_init(elev=22, azim=35)
        fig.suptitle(f'{shape_id} | seed={args.seed} | blue: FPS / green: neighbors / orange: center')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170, bbox_inches='tight')
        plt.close(fig)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(parse_args())
