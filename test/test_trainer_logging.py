"""Tests for Trainer progress bar / log artifacts (training curve figure + history files)."""
import json
import os

import pytest

torch = pytest.importorskip("torch")

from neuralrnn import AutoConfig, AutoModel, Trainer, TrainingArguments, SupervisedObjective
from neuralrnn.data import BaseDataset


class _ToyTask(BaseDataset):
    kind = "neurogym"

    def __init__(self, input_dim=3, n_actions=2, T=20, B=8):
        self.input_dim, self.output_dim, self.T, self.B = input_dim, n_actions, T, B

    def sample_batch(self):
        x = torch.randn(self.B, self.T, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.B, self.T))
        return {"inputs": x, "targets": y, "mask": None}


def _eval_fn(model):
    return {"accuracy": 0.6, "err": 0.3}


def _make_trainer(tmp_path, **kw):
    cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
    args = TrainingArguments(max_steps=30, log_every=10, eval_every=15,
                             disable_progress_bar=True, **kw)
    return Trainer(AutoModel.from_config(cfg), _ToyTask(),
                   SupervisedObjective("classification"), args, eval_fn=_eval_fn)


def test_log_artifacts_with_output_dir(tmp_path):
    trainer = _make_trainer(tmp_path, output_dir=str(tmp_path / "out"))
    hist = trainer.train()
    assert len(hist) == 30
    log_dir = tmp_path / "out"  # log artifacts go directly into output_dir
    assert (log_dir / "training_curves.png").is_file()
    assert (log_dir / "history.json").is_file()
    recs = [json.loads(l) for l in (log_dir / "history.jsonl").read_text().strip().split("\n")]
    eval_recs = [r for r in recs if r.get("eval")]
    assert len(eval_recs) == 1 and "accuracy" in eval_recs[0]
    assert len(json.loads((log_dir / "history.json").read_text())) == 30


def test_log_dir_defaults_to_temp_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _make_trainer(tmp_path)
    assert trainer.log_dir == os.path.join(".", "temp", "log")
    trainer.train()
    assert (tmp_path / "temp" / "log" / "training_curves.png").is_file()


def test_log_every_zero_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = AutoConfig.for_model("ctrnn", input_dim=3, latent_dim=16, output_dim=2)
    hist = Trainer(AutoModel.from_config(cfg), _ToyTask(), SupervisedObjective("classification"),
                   TrainingArguments(max_steps=5, log_every=0,
                                     disable_progress_bar=True)).train()
    assert len(hist) == 5
    assert not (tmp_path / "temp").exists()


def test_checkpoint_falls_back_to_outputs(tmp_path, monkeypatch):
    """output_dir=None → checkpoints go to ./outputs."""
    monkeypatch.chdir(tmp_path)
    trainer = _make_trainer(tmp_path, save_every=15)
    assert trainer.output_dir == "./outputs"
    path = trainer.save_checkpoint(15)
    assert path.startswith("./outputs") and os.path.isdir(path)
