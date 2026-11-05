"""CPU regression checks using synthetic inputs, without dataset downloads."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Pointnet_Pointnet2_pytorch/models'))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dgdata = load('dgdata', 'dgcnn/pytorch/data.py')
dgmodel = load('dgmodel', 'dgcnn/pytorch/model.py')
pnloader = load('pnloader', 'Pointnet_Pointnet2_pytorch/data_utils/ModelNetDataLoader.py')
pnmodel = load('pnmodel', 'Pointnet_Pointnet2_pytorch/models/pointnet_cls.py')


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'dataset with spaces'
        (self.root / 'night_stand').mkdir(parents=True)
        (self.root / 'modelnet40_shape_names.txt').write_text('night_stand\n', encoding='utf-8')
        for split in ('train', 'test'):
            (self.root / f'modelnet40_{split}.txt').write_text('night_stand_0001\n', encoding='utf-8')
        np.savetxt(self.root / 'night_stand/night_stand_0001.txt', np.random.default_rng(0).normal(size=(32, 6)), delimiter=',')

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_env_override_and_text_shapes(self):
        with patch.dict(os.environ, {'MODELNET40_ROOT': str(self.root)}):
            cloud, label = dgdata.ModelNet40(16, 'test')[0]
            self.assertEqual(cloud.shape, (16, 3))
            self.assertEqual(label.tolist(), [0])
            with self.assertRaises(FileNotFoundError):
                dgdata.ModelNet40(16, data_root=self.root / 'missing')
        args = SimpleNamespace(num_point=16, use_uniform_sample=False, use_normals=False, num_category=40)
        points, label = pnloader.ModelNetDataLoader(self.root, args, split='test')[0]
        self.assertEqual(points.shape, (16, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertLessEqual(np.linalg.norm(points, axis=1).max(), 1.00001)
        self.assertEqual(int(label), 0)

    def test_bad_dataset_and_short_cloud(self):
        with self.assertRaises(ValueError):
            dgdata.ModelNet40(16, 'validation', self.root)
        with self.assertRaises(ValueError):
            dgdata.ModelNet40(64, 'test', self.root)[0]

    def test_cpu_graph_and_single_item_inference(self):
        args = SimpleNamespace(k=4, emb_dims=32, dropout=0.0)
        x = torch.randn(1, 3, 16)
        self.assertEqual(tuple(dgmodel.get_graph_feature(x, k=4).shape), (1, 6, 16, 4))
        for model in (dgmodel.DGCNN(args), dgmodel.PointNet(args)):
            with torch.no_grad():
                result = model.eval()(x)
            self.assertEqual(tuple(result.shape), (1, 40))
            self.assertTrue(torch.isfinite(result).all())

    def test_pointnet_loss_optional_transform(self):
        pred = torch.log_softmax(torch.randn(2, 40, requires_grad=True), dim=1)
        for transform in (None, torch.eye(64).repeat(2, 1, 1)):
            loss = pnmodel.get_loss()(pred, torch.tensor([0, 1]), transform)
            self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_pointnet_msg_cpu_forward(self):
        model_module = load('pnmsg', 'Pointnet_Pointnet2_pytorch/models/pointnet2_cls_msg.py')
        model = model_module.get_model(40, normal_channel=False).eval()
        with torch.no_grad():
            logits, features = model(torch.randn(1, 3, 128))
        self.assertEqual(tuple(logits.shape), (1, 40))
        self.assertTrue(torch.isfinite(logits).all())

    def test_cli_help(self):
        for script in ('Pointnet_Pointnet2_pytorch/train_classification.py', 'Pointnet_Pointnet2_pytorch/test_classification.py',
                       'Pointnet_Pointnet2_pytorch/data_utils/ModelNetDataLoader.py', 'dgcnn/pytorch/main.py',
                       'dgcnn/pytorch/data.py', 'visualization_pointnet2_sampling.py'):
            result = subprocess.run([sys.executable, str(ROOT / script), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--data_root', result.stdout)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
