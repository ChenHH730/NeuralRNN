"""Apply the env-free notebook rewrites (unified trial-aligned dataset interface).

Replaces the env-driven cells in notebooks 01/04/05/09/10 with the CognitiveTaskDataset-style
interface (inputs/targets/mask/conditions), and inserts a neurogym raw-env section into
cognitive_tasks.ipynb. Edited code cells have their outputs cleared (notebooks need a re-run).
Safety: each replacement asserts a marker in the old source before writing.

Run: D:/Anaconda/envs/chh_3_11/python.exe apply_notebook_edits.py
"""
import json

NB_DIR = r"D:\phd\neuroscience\RNN\NeuralRNN\notebook"


def S(text):
    """Notebook source field: list of lines (keepends)."""
    return text.strip("\n").splitlines(keepends=True)


# ============================================================ notebook 01
NB01_C7 = """\
# Run a few trials and visualize input/output (trial-aligned dataset interface)
n_show = 2
ds_viz = load_dataset('perceptual_decision_making', batch_size=16, n_trials=n_show, dt=100)
colors = plt.cm.Set1(np.linspace(0, 1, n_show))

fig, axes = plt.subplots(ds.input_dim + 1, 1, figsize=(4, 3), sharex=True)
trial_labels = []
for i in range(n_show):
    cond = ds_viz.conditions[i]
    n = cond['n_steps']                    # true (unpadded) length of this trial
    ob = ds_viz.inputs[i, :n].numpy()      # (T, input_dim)
    gt = ds_viz.targets[i, :n].numpy()     # (T,)
    t = np.arange(n) * ds_viz.dt / 1000    # convert to seconds

    # Plot each input channel as a separate subplot
    for ch in range(ds.input_dim):
        axes[ch].plot(t, ob[:, ch], color=colors[i], alpha=0.8, lw=1.2)

    # Plot ground truth (target) at bottom
    axes[-1].plot(t, gt, color=colors[i], alpha=0.8, lw=1.2)

    trial_labels.append(f"trial {i}: groundtruth={cond.get('ground_truth', '?')}, coh={cond.get('coh', 0)}")

# Label input channel subplots
channel_names = ['Fixation', 'Stim-L', 'Stim-R']
for ch in range(ds.input_dim):
    axes[ch].set_ylabel(channel_names[ch] if ch < len(channel_names) else f'In {ch}')
    axes[ch].tick_params(labelsize=9)
axes[-1].set_ylabel('Target')
axes[-1].set_xlabel('Time (s)')
axes[-1].tick_params(labelsize=9)

fig.legend(trial_labels, bbox_to_anchor=(1, 0), fontsize=8, frameon=False, ncols=3)
fig.suptitle('Sample trials: inputs (top 3 rows) and target output (bottom)', fontsize=10)
plt.tight_layout()
plt.show()
"""

NB01_C12 = """\
# Collect per-trial activity, ground truth, coherence, and predicted actions.
# Trial-aligned dataset: pre-generated whole trials with the same interface as the
# built-in cognitive tasks (inputs/targets/mask/conditions).
num_trial = 500
ds_analysis = load_dataset('perceptual_decision_making', batch_size=16, n_trials=num_trial, dt=100)

activity_dict = {}
trial_infos = {}
action_dict = {}

model.eval()
with torch.no_grad():
    out = model(ds_analysis.inputs)   # batched forward over all trials
states_all = out.states.numpy()       # (N, T, latent_dim)
logits_all = out.outputs.numpy()      # (N, T, output_dim)

for i, cond in enumerate(ds_analysis.conditions):
    n = cond['n_steps']               # true length of this trial (unpadded)
    activity_dict[i] = states_all[i, :n]
    trial_infos[i] = cond
    # argmax()-1: neurogym actions are 1-indexed (0=fix, 1=choice0, 2=choice1),
    # ground_truth is 0-indexed (0 or 1)
    action_dict[i] = logits_all[i, n - 1].argmax() - 1

# Build DataFrame and compute accuracy
df_trials = pd.DataFrame(trial_infos).T
df_trials['action'] = pd.Series(action_dict)
acc = (df_trials['action'] == df_trials['ground_truth']).mean()
print(f'Accuracy: {acc:.3f}')
print(f'Trial info example: {trial_infos[0]}')
print(f'Unique coherences: {sorted(df_trials["coh"].unique())}')

# Concatenate all activity for PCA fitting
activity_all = np.concatenate([activity_dict[i] for i in range(num_trial)], axis=0)
print(f'Activity shape for PCA: {activity_all.shape}')
"""

NB01_C19 = """\
# Load the DelayComparison task via the registered dataset
timing = {'delay': ('choice', [200, 400, 800, 1600, 3200]),
          'response': ('constant', 500)}

ds_pw = load_dataset('delay_comparison', timing=timing,
                     batch_size=16, seq_len=100, dt=100)
print('input_dim =', ds_pw.input_dim, '| n_actions =', ds_pw.output_dim)

# Visualize sample trials (trial-aligned dataset interface)
n_show = 2
ds_viz = load_dataset('delay_comparison', timing=timing, batch_size=16, n_trials=n_show, dt=100)
fig, axes = plt.subplots(ds_pw.input_dim + 1, 1, figsize=(8, 4), sharex=True)
colors = plt.cm.Set1(np.linspace(0, 1, n_show))
trial_labels = []
for i in range(n_show):
    cond = ds_viz.conditions[i]
    n = cond['n_steps']
    ob = ds_viz.inputs[i, :n].numpy()
    gt = ds_viz.targets[i, :n].numpy()
    t = np.arange(n) * ds_viz.dt / 1000
    for ch in range(ds_pw.input_dim):
        axes[ch].plot(t, ob[:, ch], color=colors[i], alpha=0.8, lw=1.2)
    axes[-1].plot(t, gt, color=colors[i], alpha=0.8, lw=1.2)
    trial_labels.append(f'trial {i}: v1={cond.get("v1", "?")}, v2={cond.get("v2", "?")}')

for ch in range(ds_pw.input_dim):
    axes[ch].set_ylabel(f'Stim {ch}')
axes[-1].set_ylabel('Target')
axes[-1].set_xlabel('Time (s)')
fig.legend(trial_labels, loc='upper right', fontsize=8)
fig.suptitle('Parametric Working Memory: DelayComparison task', fontsize=10)
plt.tight_layout()
plt.show()
"""

NB01_C23 = """\
# Create dataset with fixed long delay for analysis (trial-aligned)
timing_analysis = {'delay': ('constant', 2000), 'response': ('constant', 500)}
ds_analysis = load_dataset('delay_comparison', timing=timing_analysis,
                           batch_size=16, n_trials=100, dt=100)

# Collect activity during delay period
num_trial_pw = len(ds_analysis)
activity_dict_pw = {}
trial_infos_pw = {}

model_pw.eval()
with torch.no_grad():
    out = model_pw(ds_analysis.inputs)   # batched forward over all trials
states_all = out.states.numpy()

for i, cond in enumerate(ds_analysis.conditions):
    # Extract activity during delay period only (epoch bounds from conditions)
    s, e = cond['epochs']['delay']
    activity_dict_pw[i] = states_all[i, s:e]
    trial_infos_pw[i] = cond

# Concatenate delay-period activity for PCA
activity_pw = np.concatenate([activity_dict_pw[i] for i in range(num_trial_pw)], axis=0)
print('Shape of delay-period neural activity:', activity_pw.shape)

# Print sample trial info
for i in range(5):
    print(f'Trial {i}:', trial_infos_pw[i])
"""

# ============================================================ notebook 04
NB04_C5 = """\
# Visualize sample trials (trial-aligned dataset interface)
n_show = 3
ds_viz = load_dataset('perceptual_decision_making', batch_size=16, n_trials=n_show,
                      dt=20, timing=timing)
colors = plt.cm.Set1(np.linspace(0, 1, n_show))

fig, axes = plt.subplots(ds.input_dim + 1, 1, figsize=(8, 6), sharex=True)
trial_labels = []
for i in range(n_show):
    cond = ds_viz.conditions[i]
    n = cond['n_steps']
    ob = ds_viz.inputs[i, :n].numpy()
    gt = ds_viz.targets[i, :n].numpy()
    t = np.arange(n) * ds_viz.dt / 1000
    for ch in range(ds.input_dim):
        axes[ch].plot(t, ob[:, ch], color=colors[i], alpha=0.8, lw=1.2)
    axes[-1].plot(t, gt, color=colors[i], alpha=0.8, lw=1.2)
    trial_labels.append(f"trial {i}: gt={cond.get('ground_truth', '?')}, coh={cond.get('coh', 0)}")

channel_names = ['Fixation', 'Stim-L', 'Stim-R']
for ch in range(ds.input_dim):
    axes[ch].set_ylabel(channel_names[ch])
axes[-1].set_ylabel('Target')
axes[-1].set_xlabel('Time (s)')
fig.legend(trial_labels, loc='upper right', fontsize=8)
fig.suptitle('Sample trials: inputs and target output', fontsize=10)
plt.tight_layout()
plt.show()
"""

NB04_C7 = """\
# Create E-I RNN with 50 units (40 excitatory, 10 inhibitory)
hidden_size = 50
cfg = AutoConfig.for_model('ei_rnn',
                           input_dim=ds.input_dim,
                           latent_dim=hidden_size,
                           output_dim=ds.output_dim,
                           dt=ds.dt,
                           sigma_rec=0.15,
                           nonlinearity_mode='post_blend')  # Match reference: f((1-α)z + α·pre)
model = AutoModel.from_config(cfg)
print(model)
print(f'\\nExcitatory units: {model.e_size}, Inhibitory units: {model.i_size}')
print(f'Readout from E units only: {cfg.readout_e_only}')
print(f'Total parameters: {model.num_parameters()}')
"""

NB04_C11 = """\
# Collect activity with fixed timing (constant 500 ms fixation/stimulus) for analysis.
# Trial-aligned dataset: whole pre-generated trials + per-trial conditions with epoch bounds.
ds_analysis = load_dataset('perceptual_decision_making', batch_size=16, n_trials=500, dt=20,
                           timing={'fixation': ('constant', 500),
                                   'stimulus': ('constant', 500)})

num_trial = len(ds_analysis)
activity_dict = {}
trial_infos = {}
stim_activity = [[], []]  # response for ground-truth 0 and 1

model.eval()
with torch.no_grad():
    out = model(ds_analysis.inputs)   # batched forward over all trials (already batch-first)
states_all = out.states.numpy()       # (N, T, latent_dim)
outputs_all = out.outputs.numpy()     # (N, T, output_dim)

for i, cond in enumerate(ds_analysis.conditions):
    n = cond['n_steps']               # true length of this trial (unpadded)
    rnn_activity = states_all[i, :n]
    activity_dict[i] = rnn_activity

    # Compute performance from the last step of the trial
    choice = int(np.argmax(outputs_all[i, n - 1]))
    correct = bool(choice == int(ds_analysis.targets[i, n - 1]))
    trial_infos[i] = {**cond, 'correct': correct, 'choice': choice}

    # Compute stimulus selectivity (epoch bounds from conditions)
    s, e = cond['epochs']['stimulus']
    stim_activity[cond['ground_truth']].append(rnn_activity[s:e])

acc = np.mean([val['correct'] for val in trial_infos.values()])
print(f'Average performance: {acc:.3f}')
"""

# ============================================================ notebook 05
NB05_C24 = """\
# Collect CTRNN hidden-state activity and task inputs over complete trials
# (trial-aligned dataset interface — same as the built-in cognitive tasks)
num_trial = 1000
ds_collect = load_dataset('perceptual_decision_making', batch_size=16, n_trials=num_trial, dt=100)

ctrnn.eval()
with torch.no_grad():
    out = ctrnn(ds_collect.inputs.to(device))   # batched forward over all trials
states_all = out.states.cpu().numpy()           # (N, T, 50)

acts, inputs, meta = [], [], []
for i, cond in enumerate(ds_collect.conditions):
    n = cond['n_steps']                         # true length of this trial (unpadded)
    acts.append(states_all[i, :n])              # (T, 50)
    inputs.append(ds_collect.inputs[i, :n].numpy())  # (T, 3)

    coh = cond.get('coh', 0)
    if hasattr(coh, '__iter__'):
        coh = float(np.mean(list(coh)))
    meta.append({
        'trial': i,
        'length': n,
        'ground_truth': int(cond['ground_truth']),
        'coh': float(coh),
    })

# Preserve trial structure: (n_trial, trial_length, n_variable)
activity_trials = np.stack(acts, axis=0)      # (1000, T, 50)
input_trials = np.stack(inputs, axis=0)       # (1000, T, 3)
meta_df = pd.DataFrame(meta)
print('Collected activity trials:', activity_trials.shape)
print('Collected input trials:  ', input_trials.shape)
print('Trials:', len(meta_df))
print(meta_df.head())
"""

# ============================================================ notebook 09
NB09_C3 = """\
ds = load_dataset('perceptual_decision_making', batch_size=16, seq_len=100, dt=100)
print('input_dim =', ds.input_dim, '| n_actions =', ds.output_dim)

# visualize a few sample trials (trial-aligned dataset interface)
n_show = 3
ds_viz = load_dataset('perceptual_decision_making', batch_size=16, n_trials=n_show, dt=100)
colors = plt.cm.Set1(np.linspace(0, 1, n_show))

fig, axes = plt.subplots(ds.input_dim + 1, 1, figsize=(6, 4), sharex=True)
trial_labels = []
for i in range(n_show):
    cond = ds_viz.conditions[i]
    n = cond['n_steps']
    ob = ds_viz.inputs[i, :n].numpy()
    gt = ds_viz.targets[i, :n].numpy()
    t = np.arange(n) * ds_viz.dt / 1000
    for ch in range(ds.input_dim):
        axes[ch].plot(t, ob[:, ch], color=colors[i], alpha=0.8, lw=1.2)
    axes[-1].plot(t, gt, color=colors[i], alpha=0.8, lw=1.2)
    trial_labels.append(f"trial {i}: gt={cond.get('ground_truth', '?')}, coh={cond.get('coh', 0)}")

channel_names = ['Fixation', 'Stim-L', 'Stim-R']
for ch in range(ds.input_dim):
    axes[ch].set_ylabel(channel_names[ch] if ch < len(channel_names) else f'In {ch}')
axes[-1].set_ylabel('Target')
axes[-1].set_xlabel('Time (s)')
fig.legend(trial_labels, bbox_to_anchor=(1, 0), fontsize=8, frameon=False, ncols=3)
fig.suptitle('Perceptual decision-making: inputs and target', fontsize=10)
plt.tight_layout()
plt.show()
"""

NB09_C12 = """\
def collect_activity_and_accuracy(model, dataset):
    '''Evaluate model on a trial-aligned dataset; return accuracy, trial info, and activity.'''
    model.eval()
    with torch.no_grad():
        out = model(dataset.inputs)       # batched forward over all trials
    states_all = out.states.numpy()       # (N, T, latent_dim)
    logits_all = out.outputs.numpy()      # (N, T, output_dim)
    infos, activities, predictions = [], {}, {}
    for i, cond in enumerate(dataset.conditions):
        n = cond['n_steps']               # true length of this trial (unpadded)
        pred = int(logits_all[i, n - 1].argmax()) - 1  # 1=choice0, 2=choice1 -> 0/1 ground_truth
        infos.append(dict(cond))
        activities[i] = states_all[i, :n]
        predictions[i] = pred
    df = pd.DataFrame(infos)
    df['action'] = pd.Series(predictions)
    acc = float((df['action'] == df['ground_truth']).mean())
    return acc, df, activities


ds_eval = load_dataset('perceptual_decision_making', batch_size=16, n_trials=500, dt=100)
acc_esn, df_esn, acts_esn = collect_activity_and_accuracy(model_esn, ds_eval)
print(f'Critical ESN accuracy (emax=0.998): {acc_esn:.3f}')
print('Unique coherences:', sorted(df_esn['coh'].unique()))
"""

NB09_C18 = """\
EMAX_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
MAX_STEPS = 1500

results = {}  # emax -> accuracy

for emax in EMAX_VALUES:
    print(f'Training readout-only ESN with emax={emax}')
    torch.manual_seed(42)  # same readout/input initialization for every emax
    cfg = AutoConfig.for_model(
        'ctrnn',
        input_dim=ds.input_dim,
        latent_dim=LATENT_DIM,
        output_dim=ds.output_dim,
        dt=100,
        tau=100,
        nonlinearity_mode='pre_activation',
        freeze_input=True,
        freeze_recurrent=True,
        freeze_output=False,
    )
    model = AutoModel.from_config(cfg)
    with torch.no_grad():
        model.h2h.weight.copy_(critical_init_recurrent(model.h2h.weight, emax=emax, seed=0))

    Trainer(
        model, ds, SupervisedObjective(task_type='classification'),
        TrainingArguments(max_steps=MAX_STEPS, learning_rate=1e-3, log_every=0)
    ).train()

    save_dir = f'./models/09/ctrnn_esn_emax_{emax}'
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

    acc, _, _ = collect_activity_and_accuracy(model, ds_eval)
    results[emax] = acc
    print(f'emax={emax} -> accuracy={acc:.3f}')

# add the critical model result
results[0.998] = acc_esn
print('All results:', results)
"""

# ============================================================ notebook 10
NB10_C20 = """\
# Load DelayComparison for training (variable delay) and analysis (fixed long delay)
timing_train = {'delay': ('choice', [200, 400, 800, 1600, 3200]),
                'response': ('constant', 500)}
ds_pw_train = load_dataset('delay_comparison', timing=timing_train,
                           batch_size=16, seq_len=100, dt=100)
print('Train input dim:', ds_pw_train.input_dim, 'output dim:', ds_pw_train.output_dim)

timing_analysis = {'delay': ('constant', 2000), 'response': ('constant', 500)}
ds_pw_analysis = load_dataset('delay_comparison', timing=timing_analysis,
                              batch_size=16, n_trials=200, dt=100)

# Visualize sample trials (trial-aligned dataset interface)
n_show = 2
ds_viz = load_dataset('delay_comparison', timing=timing_train, batch_size=16,
                      n_trials=n_show, dt=100)
fig, axes = plt.subplots(ds_pw_train.input_dim + 1, 1, figsize=(8, 4), sharex=True)
colors = plt.cm.Set1(np.linspace(0, 1, n_show))
trial_labels = []
for i in range(n_show):
    cond = ds_viz.conditions[i]
    n = cond['n_steps']
    ob = ds_viz.inputs[i, :n].numpy()
    gt = ds_viz.targets[i, :n].numpy()
    t = np.arange(n) * ds_viz.dt / 1000
    for ch in range(ds_pw_train.input_dim):
        axes[ch].plot(t, ob[:, ch], color=colors[i], alpha=0.8, lw=1.2)
    axes[-1].plot(t, gt, color=colors[i], alpha=0.8, lw=1.2)
    trial_labels.append(f'trial {i}: v1={cond.get("v1", "?")}, v2={cond.get("v2", "?")}')

for ch in range(ds_pw_train.input_dim):
    axes[ch].set_ylabel(f'Stim {ch}')
axes[-1].set_ylabel('Target')
axes[-1].set_xlabel('Time (s)')
fig.legend(trial_labels, loc='upper right', fontsize=8)
fig.suptitle('Parametric Working Memory: DelayComparison task', fontsize=10)
plt.tight_layout()
plt.show()
"""

NB10_C22 = """\
# Evaluate on the trial-aligned analysis dataset
def evaluate_delay_comparison(model, dataset):
    '''Classification accuracy on DelayComparison (trial-aligned dataset).'''
    model.eval()
    with torch.no_grad():
        out = model(dataset.inputs)   # batched forward over all trials
    logits_all = out.outputs.numpy()
    correct, total = 0, 0
    for i, cond in enumerate(dataset.conditions):
        pred = int(logits_all[i, cond['n_steps'] - 1].argmax())  # neurogym action index
        correct += (pred == cond['ground_truth'])
        total += 1
    return correct / total
"""

NB10_C24 = """\
# Collect delay-period activity for PCA (from the trial-aligned analysis dataset)
num_trial_pw = 100
activity_dict_pw = {}
trial_infos_pw = {}

model_pw = models_pw['ctrnn']
model_pw.eval()
with torch.no_grad():
    out = model_pw(ds_pw_analysis.inputs)   # batched forward over all trials
states_all = out.states.numpy()

for i, cond in enumerate(ds_pw_analysis.conditions[:num_trial_pw]):
    # Extract activity during delay period only (epoch bounds from conditions)
    s, e = cond['epochs']['delay']
    activity_dict_pw[i] = states_all[i, s:e]
    trial_infos_pw[i] = cond

activity_pw = np.concatenate([activity_dict_pw[i] for i in range(num_trial_pw)], axis=0)
pca_pw = fit_pca(activity_pw, n_components=2)
print('Explained variance ratio:', np.round(pca_pw.explained_variance_ratio, 3))

# Plot CTRNN delay-period trajectories colored by ground truth
fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(5, 2.5))
for i in range(num_trial_pw):
    trial = trial_infos_pw[i]
    activity_pc = pca_pw.transform(activity_dict_pw[i])
    color = 'red' if trial['ground_truth'] == 1 else 'blue'
    axes[0].plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color,
                 markersize=1, lw=0.5, alpha=0.3)
    if i < 1:
        axes[1].plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color,
                     markersize=2, lw=1)
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].set_title('All trials')
axes[1].set_title('Single trial')
plt.suptitle('CTRNN delay-period PCA')
plt.tight_layout()
plt.show()
"""

# ============================================================ cognitive_tasks.ipynb new cells
CT_MD = """\
## 9. Neurogym tasks: unified interface and raw env access

[Neurogym](https://github.com/neurogym/neurogym) is an external library of cognitive-task
environments (optional dependency: `pip install -e '.[neurogym]'`). Any task registered in the
installed neurogym loads through the same `load_dataset` entry point — pass the env id
(case-insensitive, `-v0` optional); `neuralrnn.data.list_neurogym_datasets()` lists what is
available. Two registered shortcuts are `perceptual_decision_making` and `delay_comparison`.

With `n_trials=...`, a neurogym dataset pre-generates complete trials and exposes **the same
interface as the built-in tasks above** (`inputs` / `targets` / `mask` / `conditions`,
`get_all_trials()`, `len(ds)`): shorter trials are zero-padded, `mask` marks valid steps, and each
condition dict carries the native trial info plus `epochs` (period bounds) and `n_steps`. Without
`n_trials`, the dataset streams fixed-length windows for training (used by notebooks 01/04/05/...).

For advanced use (RL-style interaction, timing modification, neurogym wrappers), the raw
environment remains available as `ds.env` — an unwrapped `TrialEnv` supporting `new_trial()`,
`ob`, `gt`, `trial`, `dt`, `start_ind` / `end_ind`.
"""

CT_CODE = """\
# Any neurogym env id loads through load_dataset (dynamic passthrough)
from neuralrnn.data import list_neurogym_datasets
print(f'{len(list_neurogym_datasets())} neurogym envs available, e.g.:',
      list_neurogym_datasets()[:5])

# Trial-aligned mode: same interface as the built-in cognitive tasks
ds_ng = load_dataset('PerceptualDecisionMaking-v0', n_trials=4, batch_size=4, seed=0)
print('inputs:', tuple(ds_ng.inputs.shape), '| targets:', tuple(ds_ng.targets.shape),
      '| mask:', tuple(ds_ng.mask.shape), '| len:', len(ds_ng))
print('condition keys:', sorted(ds_ng.conditions[0]))
print('epochs of trial 0:', ds_ng.conditions[0]['epochs'])

# Raw env access is still available for advanced use (RL-style stepping, timing edits)
env = ds_ng.env
env.new_trial()
print('\\nraw env: ob', env.ob.shape, '| gt', env.gt.shape, '| dt =', env.dt,
      '| trial:', env.trial)
"""

# ============================================================ apply

EDITS = {
    "01_ctrnn_fixedpoints_paradigmA.ipynb": {
        7: ("new_trial", NB01_C7),
        12: ("new_trial", NB01_C12),
        19: ("new_trial", NB01_C19),
        23: ("new_trial", NB01_C23),
    },
    "04_EIRNN_paradigmA.ipynb": {
        5: ("new_trial", NB04_C5),
        7: ("env.dt", NB04_C7),
        11: ("new_trial", NB04_C11),
    },
    "05_plrnn_reconstruction_paradigmB.ipynb": {
        24: ("new_trial", NB05_C24),
    },
    "09_echo_state_network_paradigmA.ipynb": {
        3: ("new_trial", NB09_C3),
        12: ("new_trial", NB09_C12),
        18: ("ds.env", NB09_C18),
    },
    "10_constrained_RNN_paradigmA.ipynb": {
        20: ("new_trial", NB10_C20),
        22: ("new_trial", NB10_C22),
        24: ("new_trial", NB10_C24),
    },
}


def main():
    for nb_name, cells in EDITS.items():
        path = f"{NB_DIR}/{nb_name}"
        data = json.load(open(path, encoding="utf-8"))
        for idx, (marker, new_src) in cells.items():
            cell = data["cells"][idx]
            old = "".join(cell["source"])
            assert cell["cell_type"] == "code", f"{nb_name} cell {idx} not code"
            assert marker in old, f"{nb_name} cell {idx} missing marker {marker!r}"
            cell["source"] = S(new_src)
            cell["outputs"] = []
            cell["execution_count"] = None
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{nb_name}: rewrote cells {sorted(cells)}")

    # cognitive_tasks.ipynb: insert neurogym section before the Summary cell
    path = f"{NB_DIR}/cognitive_tasks.ipynb"
    data = json.load(open(path, encoding="utf-8"))
    summary_idx = next(i for i, c in enumerate(data["cells"])
                       if c["cell_type"] == "markdown" and "## Summary" in "".join(c["source"]))
    md_cell = {"cell_type": "markdown", "id": "neurogym-env-md", "metadata": {},
               "source": S(CT_MD)}
    code_cell = {"cell_type": "code", "id": "neurogym-env-code", "metadata": {},
                 "execution_count": None, "outputs": [], "source": S(CT_CODE)}
    data["cells"][summary_idx:summary_idx] = [md_cell, code_cell]
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"cognitive_tasks.ipynb: inserted neurogym section before cell {summary_idx}")


if __name__ == "__main__":
    main()
