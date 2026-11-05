"""Run tiny CPU train/save/evaluate cycles on synthetic data in a temporary folder."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def run(argv, cwd):
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUTF8='1')
    result = subprocess.run([sys.executable, '-B', *map(str, argv)], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding='utf-8', timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    print('PASS:', Path(argv[0]).name, ' '.join(map(str, argv[1:3])))


def main():
    with tempfile.TemporaryDirectory(prefix='pointcloud-smoke-') as temp:
        work = Path(temp)
        data = work / 'data with spaces'
        data.mkdir()
        names = [f'class_{i:02d}' for i in range(40)]
        (data / 'modelnet40_shape_names.txt').write_text('\n'.join(names) + '\n')
        rng = np.random.default_rng(7)
        for split, suffix in [('train', '0001'), ('test', '0002')]:
            ids = []
            for name in names:
                (data / name).mkdir(exist_ok=True)
                shape_id = name + '_' + suffix
                ids.append(shape_id)
                np.savetxt(data / name / (shape_id + '.txt'), rng.normal(size=(32, 6)), delimiter=',')
            (data / f'modelnet40_{split}.txt').write_text('\n'.join(ids) + '\n')

        pn = work / 'pointnet'
        pn.mkdir()
        source = ROOT / 'Pointnet_Pointnet2_pytorch'
        for name in ('train_classification.py', 'test_classification.py', 'provider.py'):
            shutil.copy2(source / name, pn / name)
        for name in ('models', 'data_utils'):
            shutil.copytree(source / name, pn / name, ignore=shutil.ignore_patterns('__pycache__'))
        common = ['--data_root', data, '--use_cpu', '--num_point', '32', '--batch_size', '20', '--workers', '0', '--log_dir', 'smoke']
        run([pn / 'train_classification.py', *common, '--epoch', '1'], pn)
        assert (pn / 'log/classification/smoke/checkpoints/best_model.pth').is_file()
        run([pn / 'test_classification.py', *common, '--num_votes', '1'], pn)

        dg = ROOT / 'dgcnn/pytorch/main.py'
        common = ['--data_root', data, '--no_cuda', '--num_points', '32', '--k', '4', '--emb_dims', '32',
                  '--workers', '0', '--batch_size', '20', '--test_batch_size', '13', '--exp_name', 'smoke']
        run([dg, *common, '--epochs', '1', '--use_sgd', 'false'], work)
        checkpoint = work / 'checkpoints/smoke/models/model.t7'
        assert checkpoint.is_file()
        run([dg, *common, '--eval', '--model_path', checkpoint], work)
        legacy = work / 'legacy_parallel.t7'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        torch.save({'module.' + key: value for key, value in state.items()}, legacy)
        run([dg, *common, '--eval', '--model_path', legacy], work)
    print('All CPU training/checkpoint smoke checks passed. Synthetic accuracy is not a benchmark.')


if __name__ == '__main__':
    main()
