"""Verify the env-free (unified-interface) notebook cell logic against saved checkpoints.

Replicates the rewritten cells planned for notebooks 01/04/05/09/10 and, where a trained
checkpoint exists, compares the new trial-aligned code path against the original env-driven one.

Run: D:/Anaconda/envs/chh_3_11/python.exe verify_notebook_cells.py   (notebook env: neurogym 1.0.8)
"""
import sys

sys.path.insert(0, r"D:\phd\neuroscience\RNN\NeuralRNN\src")
NB = r"D:\phd\neuroscience\RNN\NeuralRNN\notebook"

import numpy as np
import pandas as pd
import torch

from neuralrnn import AutoConfig, AutoModel, load_dataset

torch.manual_seed(0)


# ---------------------------------------------------------------- notebook 01, cell 12 (PDM collect)
def check_nb01():
    print("== nb01 cell 12 (PDM, 500 trials, models/01/ctrnn)")
    model = AutoModel.from_pretrained(f"{NB}/models/01/ctrnn")
    model.eval()

    # NEW: trial-aligned unified interface
    ds_analysis = load_dataset("perceptual_decision_making", batch_size=16, n_trials=500,
                               dt=100, seed=0)
    num_trial = len(ds_analysis)
    with torch.no_grad():
        out = model(ds_analysis.inputs)
    states_all = out.states.numpy()
    logits_all = out.outputs.numpy()
    activity_dict, trial_infos, action_dict = {}, {}, {}
    for i, cond in enumerate(ds_analysis.conditions):
        n = cond["n_steps"]
        activity_dict[i] = states_all[i, :n]
        trial_infos[i] = cond
        action_dict[i] = logits_all[i, n - 1].argmax() - 1
    df_trials = pd.DataFrame(trial_infos).T
    df_trials["action"] = pd.Series(action_dict)
    acc_new = (df_trials["action"] == df_trials["ground_truth"]).mean()
    activity_all = np.concatenate([activity_dict[i] for i in range(num_trial)], axis=0)
    lengths = {c["n_steps"] for c in ds_analysis.conditions}
    print(f"   unique trial lengths: {lengths} (T_max={ds_analysis.inputs.shape[1]})")
    print(f"   NEW acc={acc_new:.3f}  activity_all={activity_all.shape}  cohs={sorted(df_trials['coh'].unique())}")

    # OLD: env-driven path (for comparison)
    env = load_dataset("perceptual_decision_making", batch_size=16, seq_len=100, dt=100).env
    correct = 0
    with torch.no_grad():
        for _ in range(500):
            env.new_trial()
            ob = env.ob
            out1 = model(torch.from_numpy(ob[np.newaxis]).float())
            correct += (out1.outputs[0, -1].numpy().argmax() - 1 == env.trial["ground_truth"])
    print(f"   OLD acc={correct / 500:.3f}")


# ---------------------------------------------------------------- notebook 01, cell 23 (DelayComparison delay epoch)
def check_nb01_pwm():
    print("== nb01 cell 23 (DelayComparison delay epoch, plumbing only)")
    timing_analysis = {"delay": ("constant", 2000), "response": ("constant", 500)}
    ds = load_dataset("delay_comparison", timing=timing_analysis, batch_size=16,
                      n_trials=100, dt=100, seed=0)
    cond = ds.conditions[0]
    print(f"   epochs: {cond['epochs']}")
    assert "delay" in cond["epochs"], "delay epoch missing!"
    s, e = cond["epochs"]["delay"]
    assert e - s == 20, f"delay should be 2000ms/100 = 20 steps, got {e - s}"
    cfg = AutoConfig.for_model("ctrnn", input_dim=ds.input_dim, latent_dim=16,
                               output_dim=ds.output_dim, dt=100, tau=100)
    model_pw = AutoModel.from_config(cfg).eval()
    with torch.no_grad():
        out = model_pw(ds.inputs)
    states_all = out.states.numpy()
    activity_dict_pw = {i: states_all[i, cond["epochs"]["delay"][0]:cond["epochs"]["delay"][1]]
                        for i, cond in enumerate(ds.conditions)}
    activity_pw = np.concatenate([activity_dict_pw[i] for i in range(len(ds))], axis=0)
    print(f"   delay activity: {activity_pw.shape} (expect (2000, 16))")
    assert activity_pw.shape == (2000, 16)


# ---------------------------------------------------------------- notebook 04, cells 7+11 (EI-RNN, fixed timing)
def check_nb04():
    print("== nb04 cells 7/11 (EI-RNN, fixed 500/500ms timing, fresh model)")
    ds_analysis = load_dataset("perceptual_decision_making", batch_size=16, n_trials=100, dt=20,
                               timing={"fixation": ("constant", 500),
                                       "stimulus": ("constant", 500)}, seed=0)
    cfg = AutoConfig.for_model("ei_rnn", input_dim=ds_analysis.input_dim, latent_dim=50,
                               output_dim=ds_analysis.output_dim, dt=ds_analysis.dt,
                               sigma_rec=0.15, nonlinearity_mode="post_blend")
    model = AutoModel.from_config(cfg).eval()
    with torch.no_grad():
        out = model(ds_analysis.inputs)
    states_all = out.states.numpy()
    outputs_all = out.outputs.numpy()
    activity_dict, trial_infos = {}, {}
    stim_activity = [[], []]
    for i, cond in enumerate(ds_analysis.conditions):
        n = cond["n_steps"]
        activity_dict[i] = states_all[i, :n]
        choice = int(np.argmax(outputs_all[i, n - 1]))
        correct = bool(choice == int(ds_analysis.targets[i, n - 1]))
        trial_infos[i] = {**cond, "correct": correct, "choice": choice}
        s, e = cond["epochs"]["stimulus"]
        stim_activity[cond["ground_truth"]].append(states_all[i, s:e])
    acc = np.mean([v["correct"] for v in trial_infos.values()])
    c0 = ds_analysis.conditions[0]
    print(f"   epochs: {c0['epochs']}  acc(untrained)={acc:.3f}")
    print(f"   stim groups: {[len(g) for g in stim_activity]}, stim len={stim_activity[0][0].shape}")
    assert c0["epochs"]["stimulus"] == (25, 50), "500ms fixation + 500ms stimulus at dt=20"


# ---------------------------------------------------------------- notebook 05, cell 24 (collect for PLRNN)
def check_nb05():
    print("== nb05 cell 24 (1000->100 trials, models/05/ctrnn_task)")
    ctrnn = AutoModel.from_pretrained(f"{NB}/models/05/ctrnn_task")
    ctrnn.eval()
    num_trial = 100
    ds_collect = load_dataset("perceptual_decision_making", batch_size=16, n_trials=num_trial,
                              dt=100, seed=0)
    with torch.no_grad():
        out = ctrnn(ds_collect.inputs)
    states_all = out.states.numpy()
    acts, inputs, meta = [], [], []
    for i, cond in enumerate(ds_collect.conditions):
        n = cond["n_steps"]
        acts.append(states_all[i, :n])
        inputs.append(ds_collect.inputs[i, :n].numpy())
        coh = cond.get("coh", 0)
        if hasattr(coh, "__iter__"):
            coh = float(np.mean(list(coh)))
        meta.append({"trial": i, "length": n,
                     "ground_truth": int(cond["ground_truth"]), "coh": float(coh)})
    activity_trials = np.stack(acts, axis=0)
    input_trials = np.stack(inputs, axis=0)
    meta_df = pd.DataFrame(meta)
    print(f"   activity_trials={activity_trials.shape} input_trials={input_trials.shape} meta={meta_df.shape}")


# ---------------------------------------------------------------- notebook 09, cell 12 (ESN eval helper)
def check_nb09():
    print("== nb09 cell 12 (collect_activity_and_accuracy, models/09/ctrnn_esn_emax_0.998)")

    def collect_activity_and_accuracy(model, dataset):
        """Evaluate model on a trial-aligned dataset; return accuracy, trial info, activity."""
        model.eval()
        with torch.no_grad():
            out = model(dataset.inputs)
        states_all = out.states.numpy()
        logits_all = out.outputs.numpy()
        infos, activities, predictions = [], {}, {}
        for i, cond in enumerate(dataset.conditions):
            n = cond["n_steps"]
            infos.append(dict(cond))
            activities[i] = states_all[i, :n]
            predictions[i] = int(logits_all[i, n - 1].argmax()) - 1  # 1/2 -> 0/1 ground_truth
        df = pd.DataFrame(infos)
        df["action"] = pd.Series(predictions)
        acc = float((df["action"] == df["ground_truth"]).mean())
        return acc, df, activities

    model = AutoModel.from_pretrained(f"{NB}/models/09/ctrnn_esn_emax_0.998")
    ds_eval = load_dataset("perceptual_decision_making", batch_size=16, n_trials=500,
                           dt=100, seed=0)
    acc, df, acts = collect_activity_and_accuracy(model, ds_eval)
    print(f"   NEW acc={acc:.3f}  acts={len(acts)}  cohs={sorted(df['coh'].unique())}")

    # OLD env-driven helper for comparison
    def old_collect(model, env, n_trials=500):
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(n_trials):
                env.new_trial()
                ob = env.ob
                out = model(torch.from_numpy(ob[np.newaxis]).float())
                pred = int(out.outputs[0, -1].numpy().argmax()) - 1
                correct += (pred == int(env.trial["ground_truth"]))
        return correct / n_trials

    env = load_dataset("perceptual_decision_making", batch_size=16, seq_len=100, dt=100).env
    print(f"   OLD acc={old_collect(model, env):.3f}")


# ---------------------------------------------------------------- notebook 10, cells 22/24 (DelayComparison eval + delay PCA)
def check_nb10():
    print("== nb10 cells 22/24 (models/10/delaycmp_ctrnn)")
    model_pw = AutoModel.from_pretrained(f"{NB}/models/10/delaycmp_ctrnn")
    model_pw.eval()
    timing_analysis = {"delay": ("constant", 2000), "response": ("constant", 500)}
    ds_pw_analysis = load_dataset("delay_comparison", timing=timing_analysis,
                                  batch_size=16, n_trials=200, dt=100, seed=0)

    # NEW evaluate (cell 22)
    with torch.no_grad():
        out = model_pw(ds_pw_analysis.inputs)
    logits_all = out.outputs.numpy()
    correct = sum(int(logits_all[i, c["n_steps"] - 1].argmax()) == c["ground_truth"]
                  for i, c in enumerate(ds_pw_analysis.conditions))
    acc_new = correct / len(ds_pw_analysis)
    print(f"   NEW evaluate acc={acc_new:.3f}")

    # OLD env-driven evaluate for comparison
    env = load_dataset("delay_comparison", timing=timing_analysis,
                       batch_size=16, seq_len=100, dt=100).env
    correct_old = 0
    with torch.no_grad():
        for _ in range(200):
            env.new_trial()
            ob = env.ob
            out1 = model_pw(torch.from_numpy(ob[np.newaxis]).float())
            correct_old += (out1.outputs[0, -1].argmax().item() == env.trial["ground_truth"])
    print(f"   OLD evaluate acc={correct_old / 200:.3f}")

    # NEW delay-epoch collect (cell 24)
    states_all = out.states.numpy()
    activity_dict_pw, trial_infos_pw = {}, {}
    for i, cond in enumerate(ds_pw_analysis.conditions):
        s, e = cond["epochs"]["delay"]
        activity_dict_pw[i] = states_all[i, s:e]
        trial_infos_pw[i] = cond
    activity_pw = np.concatenate([activity_dict_pw[i] for i in range(100)], axis=0)
    print(f"   delay activity (100 trials): {activity_pw.shape}, gt values: "
          f"{sorted({c['ground_truth'] for c in ds_pw_analysis.conditions})}")


if __name__ == "__main__":
    check_nb01()
    check_nb01_pwm()
    check_nb04()
    check_nb05()
    check_nb09()
    check_nb10()
    print("ALL CHECKS DONE")
