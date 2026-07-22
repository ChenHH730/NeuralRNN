"""冒烟测试 / 契约规范。

这些测试既验证框架自洽，也充当"契约的可执行说明书"。运行：
    pip install -e '.[dev]'
    pytest -q

（本仓库的开发环境若无 torch，CI 会跳过；本文件不依赖任何需下载的数据集。）
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, NeuralDynamicsModel
from neuralrnn import Trainer, TrainingArguments
from neuralrnn import SupervisedObjective, TeacherForcingObjective
from neuralrnn.data import BaseDataset


# --------- 一个最小的内存数据集 ---------
class _ToyTimeSeries(BaseDataset):
    kind = "timeseries"

    def __init__(self, N=3, T=400, B=8, L=50):
        self.N, self.B, self.L = N, B, L
        t = np.linspace(0, 20 * np.pi, T)
        self.X = torch.tensor(np.stack([np.sin(t), np.cos(t), np.sin(2 * t)][:N], 1),
                              dtype=torch.float32)
        self.input_dim = self.output_dim = N

    def sample_batch(self):
        import random
        xs = []
        for _ in range(self.B):
            s = random.randint(0, self.X.shape[0] - self.L - 2)
            xs.append(self.X[s:s + self.L])
        return {"activity": torch.stack(xs)}


class _ToyTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim=3, n_actions=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}

    def task_input(self):
        return torch.zeros(self.input_dim)


# ============================ 契约：构造 / 形状 / 存读 ============================
@pytest.mark.parametrize("model_type,kw", [
    ("ctrnn", dict(input_dim=3, latent_dim=16, output_dim=2)),
    ("shallow_plrnn", dict(input_dim=0, latent_dim=3, output_dim=3,
                           hidden_dim=20, autonomous=True)),
])
def test_construct_and_contract(model_type, kw):
    cfg = AutoConfig.for_model(model_type, **kw)
    model = AutoModel.from_config(cfg)
    assert isinstance(model, NeuralDynamicsModel)
    B, M = 4, cfg.latent_dim
    z = model.init_state(B)
    assert z.shape == (B, M)
    x = None if kw.get("autonomous") else torch.randn(B, cfg.input_dim)
    z1 = model.recurrence(x, z)
    assert z1.shape == (B, M)
    assert model.readout(z1).shape == (B, cfg.output_dim)


def test_forward_rollout_shapes():
    cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
    model = AutoModel.from_config(cfg)
    out = model(torch.randn(4, 10, 3))
    assert out.outputs.shape == (4, 10, 2)
    assert out.states.shape == (4, 10, 16)


def test_save_load_roundtrip(tmp_path):
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                               output_dim=3, hidden_dim=20, autonomous=True)
    model = AutoModel.from_config(cfg)
    model.save_pretrained(str(tmp_path))
    reloaded = AutoModel.from_pretrained(str(tmp_path))
    z = torch.randn(2, 3)
    assert torch.allclose(model.recurrence(None, z), reloaded.recurrence(None, z), atol=1e-6)


# ============================ 契约：PLRNN 解析雅可比 == 自动微分 ============================
def test_plrnn_analytic_jacobian_matches_autodiff():
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                               output_dim=3, hidden_dim=20, autonomous=True)
    model = AutoModel.from_config(cfg)
    z = torch.randn(3) * 0.3
    J_analytic = model.jacobian(z)
    J_autodiff = NeuralDynamicsModel.jacobian(model, z)   # 基类自动微分兜底
    assert torch.allclose(J_analytic, J_autodiff, atol=1e-4)
    assert model.supports_analytic_fixed_points


# ============================ 训练：两范式各跑几步 ============================
def test_train_supervised_paradigmA():
    ds = _ToyTask()
    cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
    model = AutoModel.from_config(cfg)
    hist = Trainer(model, ds, SupervisedObjective("classification"),
                   TrainingArguments(max_steps=5, log_every=0)).train()
    assert len(hist) == 5 and "loss" in hist[-1]


def test_train_gtf_paradigmB():
    ds = _ToyTimeSeries()
    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                               output_dim=3, hidden_dim=20, autonomous=True)
    model = AutoModel.from_config(cfg)
    hist = Trainer(model, ds, TeacherForcingObjective(alpha=0.2),
                   TrainingArguments(max_steps=5, log_every=0)).train()
    assert len(hist) == 5


# ============================ 分析：跑通即可（小规模）============================
def test_analysis_numeric_fixed_points():
    from neuralrnn.analysis import find_fixed_points
    cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=8, output_dim=2)
    model = AutoModel.from_config(cfg)
    fps = find_fixed_points(model, backend="numeric",
                            task_input=torch.zeros(3), n_candidates=8, n_iters=50)
    assert hasattr(fps, "points")


def test_analysis_metrics_and_lyapunov():
    from neuralrnn.analysis import (state_space_divergence, power_spectrum_error,
                                    max_lyapunov_exponent)
    true = np.random.randn(500, 3)
    gen = true + 0.01 * np.random.randn(500, 3)
    assert np.isfinite(state_space_divergence(gen, true))
    assert np.isfinite(power_spectrum_error(true, gen))

    cfg = AutoConfig.for_model("shallow_plrnn", input_dim=0, latent_dim=3,
                               output_dim=3, hidden_dim=20, autonomous=True)
    model = AutoModel.from_config(cfg)
    lam = max_lyapunov_exponent(model, torch.randn(3), T=50, T_trans=20)
    assert np.isfinite(lam)
