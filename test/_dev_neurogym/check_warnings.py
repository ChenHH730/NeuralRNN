"""Dev check: notebook 01/04 env-access patterns must produce no wrapper-attribute warnings.

Run with both conda envs:
  D:/Anaconda/envs/chh_3_11/python.exe check_warnings.py   (neurogym 1.0.8)
  D:/Anaconda/envs/exp/python.exe      check_warnings.py   (neurogym 2.3.1)
"""
import sys
import warnings

sys.path.insert(0, r"D:\phd\neuroscience\RNN\NeuralRNN\src")

import numpy as np
import torch

WRAPPER_MSG = "to get variables from other wrappers is deprecated"


def check_no_warnings(fn, label):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    bad = [w for w in rec if WRAPPER_MSG in str(w.message)]
    print(f"[{label}] warnings: total={len(rec)} wrapper-attr={len(bad)}")
    for w in bad:
        print("   !!", w.message)
    return len(bad) == 0


from neuralrnn import load_dataset  # noqa: E402
from neuralrnn.data import list_neurogym_datasets, neurogym_version  # noqa: E402

print("neurogym version:", neurogym_version())
envs = list_neurogym_datasets()
print(f"{len(envs)} envs available")


def nb01_pattern():
    """Notebook 01: PDM sample-trial viz + delay_comparison timing access."""
    ds = load_dataset('perceptual_decision_making', batch_size=4, seq_len=50, dt=100)
    env = ds.env
    for _ in range(3):
        env.new_trial()
        ob, gt = env.ob, env.gt
        _ = np.arange(ob.shape[0]) * env.dt / 1000
        _ = env.trial.get('coh', 0)
        _ = env.trial.get('ground_truth', '?')
    b = ds.sample_batch()
    assert b['inputs'].shape == (4, 50, ds.input_dim), b['inputs'].shape
    assert b['targets'].shape == (4, 50), b['targets'].shape
    assert b['targets'].dtype == torch.long

    ds_pw = load_dataset('delay_comparison',
                         timing={'delay': ('choice', [200, 400]), 'response': ('constant', 500)},
                         batch_size=4, seq_len=50, dt=100)
    env_pw = ds_pw.env
    env_pw.new_trial()
    _ = env_pw.ob, env_pw.gt, env_pw.dt
    _ = env_pw.trial.get('v1'), env_pw.trial.get('v2')
    _ = env_pw.start_ind['delay'], env_pw.end_ind['delay']


def nb04_pattern():
    """Notebook 04: custom timing, reset, timing.update, start/end_ind."""
    timing = {'fixation': ('choice', (50, 100, 200, 400)),
              'stimulus': ('choice', (100, 200, 400, 800))}
    ds = load_dataset('perceptual_decision_making', batch_size=4, seq_len=50,
                      dt=20, timing=timing)
    env = ds.env
    env.new_trial()
    _ = env.ob, env.gt, env.dt, env.trial.get('coh')
    env.reset()
    env.timing.update({'fixation': ('constant', 500), 'stimulus': ('constant', 500)})
    env.new_trial()
    _ = env.start_ind['stimulus'], env.end_ind['stimulus']


ok = check_no_warnings(nb01_pattern, "nb01 pattern")
ok &= check_no_warnings(nb04_pattern, "nb04 pattern")

# --- load_dataset passthrough ---
ds = load_dataset('GoNogo-v0', batch_size=2, seq_len=30)
print(f"passthrough 'GoNogo-v0' ok: in={ds.input_dim} out={ds.output_dim} type={ds.output_type}")
ds2 = load_dataset('gonogo', batch_size=2, seq_len=30)
print(f"passthrough 'gonogo' ok: in={ds2.input_dim} out={ds2.output_dim}")
try:
    load_dataset('nonexistent_task_xyz')
    print("!! KeyError NOT raised for unknown name")
    ok = False
except KeyError:
    print("KeyError raised for unknown name: ok")

# --- internal registry entry must not be shadowed ---
from neuralrnn.data import CognitiveTaskDataset  # noqa: E402
ds_internal = load_dataset('go_nogo', n_steps=30, batch_size=2)
print(f"internal 'go_nogo' still cognitive_task: {type(ds_internal).__name__}")

# --- continuous action space ---
ds_c = load_dataset('ReachingDelayResponse-v0', batch_size=2, seq_len=30)
b_c = ds_c.sample_batch()
print(f"continuous ReachingDelayResponse-v0: targets {tuple(b_c['targets'].shape)} dtype={b_c['targets'].dtype}")
assert b_c['targets'].dtype == torch.float32
assert b_c['targets'].shape == (2, 30, ds_c.output_dim)

# --- seed reproducibility ---
a = load_dataset('perceptual_decision_making', batch_size=4, seq_len=50, seed=0)
b = load_dataset('perceptual_decision_making', batch_size=4, seq_len=50, seed=0)
same = all(torch.equal(a.sample_batch()[k], b.sample_batch()[k]) for k in ('inputs', 'targets'))
print("seed reproducible:", same)
ok &= same

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
