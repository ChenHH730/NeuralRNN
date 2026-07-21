"""Dev check: smoke-test every registered neurogym env through NeurogymDataset.

Run with both conda envs:
  D:/Anaconda/envs/chh_3_11/python.exe smoke_all_envs.py   (neurogym 1.0.8)
  D:/Anaconda/envs/exp/python.exe      smoke_all_envs.py   (neurogym 2.3.1)
"""
import sys
import warnings

sys.path.insert(0, r"D:\phd\neuroscience\RNN\NeuralRNN\src")

from neuralrnn.data.neurogym_dataset import (  # noqa: E402
    NeurogymDataset, list_neurogym_datasets, neurogym_version,
)

print("neurogym:", neurogym_version())
envs = list_neurogym_datasets()
ok, fail = [], []
for env_id in envs:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = NeurogymDataset.from_task(env_id, batch_size=2, seq_len=50)
            b = ds.sample_batch()
        ok.append((env_id, f"in={ds.input_dim}", f"out={ds.output_dim}",
                   ds.output_type, f"targets{tuple(b['targets'].shape)}"))
    except Exception as e:
        fail.append((env_id, f"{type(e).__name__}: {str(e).splitlines()[0]}"))

print(f"\nOK ({len(ok)}):")
for x in ok:
    print("  ", *x)
print(f"\nFAIL ({len(fail)}):")
for x in fail:
    print("  ", *x)
