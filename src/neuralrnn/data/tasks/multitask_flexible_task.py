"""Driscoll et al. (2024) 15-task flexible multitask dataset.

This module re-implements the 15 cognitive tasks from:
    Driscoll, L. N., Shenoy, K., and Sussillo, D. (2024).
    Flexible multitask computation in recurrent networks utilizes shared dynamical motifs.
    Nature Neuroscience, 27(7), 1349-1363.

The task set is a simplified, low-dimensional-stimulus version of Yang et al. (2019).
Each task is indicated by a one-hot rule input. Stimuli and responses are encoded as
2D circular vectors (sin, cos) instead of the 32-unit ring used in Yang et al.

Input format:
    index 0       : fixation input (1=fixate, 0=respond)
    indices 1-2   : modality 1 stimulus (A1*sin(theta1), A1*cos(theta1))
    indices 3-4   : modality 2 stimulus (A2*sin(theta2), A2*cos(theta2))
    indices 5-19  : rule input (one-hot, 15 tasks)

Target format:
    index 0       : fixation output target (0.8 during fixation, 0 during response)
    indices 1-2   : response direction (sin(phi), cos(phi))

Mask format:
    (batch, time, output_dim) float array.
    First 100 ms of each trial is a grace period (weight 0).
    Pre-response period weight 1; response period weight 5.
    Fixation output channel weight is doubled.
"""
from __future__ import division

import numpy as np

DT = 20.0
TAU = 100.0
ALPHA = DT / TAU
INPUT_DIM = 20
OUTPUT_DIM = 3
N_RULE = 15
FIXATION_TARGET = 0.8
RESPONSE_TARGET = 0.0
RULE_START = 5

RULES_ALL = [
    'fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
    'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
]
RULE_INDEX_MAP = {rule: idx for idx, rule in enumerate(RULES_ALL)}

RULE_NAME = {
    'reactgo': 'RT Go', 'delaygo': 'Dly Go', 'fdgo': 'Go',
    'delaydm1': 'Dly DM 1', 'delaydm2': 'Dly DM 2',
    'contextdelaydm1': 'Ctx Dly DM 1', 'contextdelaydm2': 'Ctx Dly DM 2',
    'multidelaydm': 'MultSen Dly DM',
    'reactanti': 'RT Anti', 'delayanti': 'Dly Anti', 'fdanti': 'Anti',
    'dmsgo': 'DMS', 'dmsnogo': 'DNMS', 'dmcgo': 'DMC', 'dmcnogo': 'DNMC',
}


def _get_dist(original_dist):
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


class Trial(object):
    """Trial container for the flexible (Driscoll 2024) task battery.

    Holds the input tensor x (tdim, batch, n_input), target tensor y
    (tdim, batch, n_output), and response locations y_loc, filled in by the
    per-rule generator functions below via the add_* helpers.
    """

    def __init__(self, config, tdim, batch_size):
        self.float_type = 'float32'
        self.config = config
        self.dt = self.config['dt']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        self.y[:, :, 0] = RESPONSE_TARGET
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)
        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var):
        """Broadcast a scalar/None parameter to a per-trial list of length batch_size."""
        if var is None:
            return [None] * self.batch_size
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Write an epoch into x/y.

        loc_type: "fix_in" (fixation input), "stim" (2-channel ring stimulus in
        modality ``mods``), "fix_out" (fixation target), "out" (ring response
        target). ons/offs/strengths/mods/locs accept scalars or per-trial lists.
        """
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)
        locs = self.expand(locs)
        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]:offs[i], i, 0] = 1
            elif loc_type == 'stim':
                start = 1 + (mods[i] - 1) * 2
                self.x[ons[i]:offs[i], i, start:start + 2] += self.add_x_loc(locs[i]) * strengths[i]
            elif loc_type == 'fix_out':
                self.y[ons[i]:offs[i], i, 0] = FIXATION_TARGET
            elif loc_type == 'out':
                self.y[ons[i]:offs[i], i, 1:] += self.add_y_loc(locs[i])
                self.y_loc[ons[i]:offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add i.i.d. Gaussian input noise (std = sigma_x * sqrt(2/alpha))."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        """Build the loss mask c_mask (tdim*batch, n_output): 0 before 100 ms,
        1 from 100 ms until pre_offs, 0 again until post_ons, then 5;
        the fixation channel is weighted 2x."""
        pre_on = int(100 / self.dt)
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)
        c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
        for i in range(self.batch_size):
            c_mask[post_ons[i]:, i, :] = 5.0
            c_mask[pre_on:pre_offs[i], i, :] = 1.0
        c_mask[:, :, 0] *= 2.0
        self.c_mask = c_mask.reshape((self.tdim * self.batch_size, self.n_output))

    def add_rule(self, rule, on=None, off=None, strength=1.0):
        """Activate the one-hot rule input channel for ``rule`` during [on, off)."""
        if isinstance(rule, int):
            ind_rule = self.config['rule_start'] + rule
        else:
            ind_rule = self.config['rule_start'] + RULE_INDEX_MAP[rule]
        self.x[on:off, :, ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Encode a ring angle as (sin, cos) stimulus channels."""
        return np.array([np.sin(x_loc), np.cos(x_loc)])

    def add_y_loc(self, y_loc):
        """Encode a ring angle as (sin, cos) response channels."""
        return np.array([np.sin(y_loc), np.cos(y_loc)])


# ---------------------------------------------------------------------------
# Delayed response family (fdgo / fdanti)
# ---------------------------------------------------------------------------

def _fdgo_core(config, mode, anti_response, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_mod = rng.choice([1, 2])
        stim_ons = int(rng.uniform(300, 700) / dt)
        fix_offs = stim_ons + int(rng.uniform(200, 1500) / dt)
        tdim = fix_offs + int(rng.uniform(300, 700) / dt)
    elif mode == 'test':
        tdim = int(2000 / dt)
        n_stim_loc, n_stim_mod = 40, 2
        batch_size = n_stim_loc * n_stim_mod
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size), (n_stim_loc, n_stim_mod))
        stim_ons = int(500 / dt)
        fix_offs = int(1500 / dt)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = ind_stim_mod + 1
    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        stim_time = int(p['stim_time'] / dt)
        batch_size = len(stim_locs)
        stim_ons = int(300 / dt)
        fix_offs = stim_ons + stim_time
        tdim = int(400 / dt) + fix_offs
        stim_mod = 1
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)
    stim_locs = np.array(stim_locs)
    response_locs = (stim_locs + np.pi) % (2 * np.pi) if anti_response else stim_locs

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=fix_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, fix_offs),
        'go1': (fix_offs, None),
    }
    return trial


def fdgo(config, mode, **kwargs):
    """Trial generator for the ``fdgo`` rule (dispatched by generate_trials)."""
    return _fdgo_core(config, mode, False, **kwargs)


def fdanti(config, mode, **kwargs):
    """Trial generator for the ``fdanti`` rule (dispatched by generate_trials)."""
    return _fdgo_core(config, mode, True, **kwargs)


# ---------------------------------------------------------------------------
# Memory response family (delaygo / delayanti)
# ---------------------------------------------------------------------------

def _delaygo_core(config, mode, anti_response, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(rng.uniform(300, 700) / dt)
        stim_offs = stim_ons + int(rng.uniform(200, 1500) / dt)
        fix_offs = stim_offs + int(rng.uniform(200, 1600) / dt)
        tdim = fix_offs + int(rng.uniform(300, 700) / dt)
        stim_mod = rng.choice([1, 2])
    elif mode == 'test':
        tdim = int(2500 / dt)
        n_stim_loc, n_stim_mod = 40, 2
        batch_size = n_stim_loc * n_stim_mod
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size), (n_stim_loc, n_stim_mod))
        fix_offs = int(2000 / dt)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_ons = int(500 / dt)
        stim_mod = ind_stim_mod + 1
        stim_offs = int(1000 / dt)
    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        stim_ons = int(p['stim_ons'] / dt)
        stim_offs = int(p['stim_offs'] / dt)
        delay_time = int(p['delay_time'] / dt)
        fix_offs = stim_offs + delay_time
        tdim = int(400 / dt) + fix_offs
        stim_mod = 1
        batch_size = len(stim_locs)
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)
    stim_locs = np.array(stim_locs)
    response_locs = (stim_locs + np.pi) % (2 * np.pi) if anti_response else stim_locs

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, stim_offs),
        'delay1': (stim_offs, fix_offs),
        'go1': (fix_offs, None),
    }
    return trial


def delaygo(config, mode, **kwargs):
    """Trial generator for the ``delaygo`` rule (dispatched by generate_trials)."""
    return _delaygo_core(config, mode, False, **kwargs)


def delayanti(config, mode, **kwargs):
    """Trial generator for the ``delayanti`` rule (dispatched by generate_trials)."""
    return _delaygo_core(config, mode, True, **kwargs)


# ---------------------------------------------------------------------------
# Reaction-time family (reactgo / reactanti)
# ---------------------------------------------------------------------------

def _reactgo_core(config, mode, anti_response, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_ons = int(rng.uniform(500, 2500) / dt)
        tdim = stim_ons + int(rng.uniform(300, 1700) / dt)
        stim_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim_mod = rng.choice([1, 2])
    elif mode == 'test':
        tdim = int(2500 / dt)
        n_stim_loc, n_stim_mod = 40, 2
        batch_size = n_stim_loc * n_stim_mod
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size), (n_stim_loc, n_stim_mod))
        stim_ons = int(2000 / dt)
        stim_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim_mod = ind_stim_mod + 1
    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        batch_size = len(stim_locs)
        stim_ons = int(1000 / dt)
        tdim = int(400 / dt) + stim_ons
        stim_mod = 1
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = stim_ons + int(100 / dt)
    stim_locs = np.array(stim_locs)
    response_locs = (stim_locs + np.pi) % (2 * np.pi) if anti_response else stim_locs

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    trial.add('fix_out', offs=stim_ons)
    trial.add('out', response_locs, ons=stim_ons)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'go1': (stim_ons, None),
    }
    return trial


def reactgo(config, mode, **kwargs):
    """Trial generator for the ``reactgo`` rule (dispatched by generate_trials)."""
    return _reactgo_core(config, mode, False, **kwargs)


def reactanti(config, mode, **kwargs):
    """Trial generator for the ``reactanti`` rule (dispatched by generate_trials)."""
    return _reactgo_core(config, mode, True, **kwargs)


# ---------------------------------------------------------------------------
# Decision-making family (delaydm / contextdelaydm / multidelaydm)
# ---------------------------------------------------------------------------

def _contextdm_genstim(batch_size, rng, stim_coh_range=None):
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.random.uniform(0, 0.8, (100,))
    stim_coh = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([+1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh * stim_sign
    stim2_strengths = stim_mean - stim_coh * stim_sign
    return stim1_strengths, stim2_strengths


def _delaydm(config, mode, stim_mod, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
        stim_coh_range = np.random.uniform(0.005, 0.8, (100,))
        if config.get('easy_task', False):
            stim_coh_range *= 2

        stims_coh = rng.choice(stim_coh_range, (batch_size,))
        stims_sign = rng.choice([1, -1], (batch_size,))
        stim1_strengths = stims_mean + stims_coh * stims_sign
        stim2_strengths = stims_mean - stims_coh * stims_sign

        stim1_ons = int(rng.uniform(200, 600) / dt)
        stim1_offs = stim1_ons + int(rng.uniform(200, 1600) / dt)
        stim2_ons = stim1_offs + int(rng.uniform(200, 1600) / dt)
        stim2_offs = stim2_ons + int(rng.uniform(200, 1600) / dt)
        fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
        tdim = fix_offs + int(rng.uniform(300, 700) / dt)

    elif mode == 'test':
        tdim = int(3000 / dt)
        n_stim_loc, n_stim1_strength = 40, 4
        batch_size = n_stim_loc * n_stim1_strength
        ind_stim_loc, ind_stim1_strength = np.unravel_index(range(batch_size), (n_stim_loc, n_stim1_strength))

        fix_offs = int(2700 / dt)
        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim2_locs = (stim1_locs + np.pi) % (2 * np.pi)
        stim1_strengths = 1.0 * ind_stim1_strength / n_stim1_strength + 0.5
        stim2_strengths = 2 - stim1_strengths
        stim1_ons = int(500 / dt)
        stim1_offs = int(1000 / dt)
        stim2_ons = int(2000 / dt)
        stim2_offs = int(2500 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_strengths = p['stim1_strengths']
        stim2_strengths = p['stim2_strengths']
        stim1_ons = int(p['stim1_ons'] / dt)
        stim1_offs = int(p['stim1_offs'] / dt)
        stim2_ons = int(p['stim2_ons'] / dt)
        stim2_offs = int(p['stim2_offs'] / dt)
        batch_size = len(stim1_locs)
        fix_offs = int(200 / dt) + stim2_offs
        tdim = int(300 / dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i] else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim1_ons),
        'stim1': (stim1_ons, stim1_offs),
        'delay1': (stim1_offs, stim2_ons),
        'stim2': (stim2_ons, stim2_offs),
        'delay2': (stim2_offs, fix_offs),
        'go1': (fix_offs, None),
    }
    return trial


def delaydm1(config, mode, **kwargs):
    """Trial generator for the ``delaydm1`` rule (dispatched by generate_trials)."""
    return _delaydm(config, mode, 1, **kwargs)


def delaydm2(config, mode, **kwargs):
    """Trial generator for the ``delaydm2`` rule (dispatched by generate_trials)."""
    return _delaydm(config, mode, 2, **kwargs)


def _contextdelaydm(config, mode, attend_mod, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stim_coh_range = np.random.uniform(0.005, 0.8, (100,))
        if config.get('easy_task', False):
            stim_coh_range *= 2

        if attend_mod in (1, 2):
            stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        else:
            stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod12_diff = stim1_strengths * np.random.uniform(0.005, 0.8, (batch_size,)) * np.random.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2

            stim2_mod12_diff = stim2_strengths * np.random.uniform(0.005, 0.8, (batch_size,)) * np.random.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

        stim1_ons = int(rng.uniform(200, 600) / dt)
        stim1_offs = stim1_ons + int(rng.uniform(200, 1600) / dt)
        stim2_ons = stim1_offs + int(rng.uniform(200, 1600) / dt)
        stim2_offs = stim2_ons + int(rng.uniform(200, 1600) / dt)
        fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
        tdim = fix_offs + int(rng.uniform(300, 700) / dt)

    elif mode == 'test':
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = 40, 4, 4
        batch_size = n_stim_loc * n_stim_mod1_strength * n_stim_mod2_strength
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(
            range(batch_size), (n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength))

        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim2_locs = (stim1_locs + np.pi) % (2 * np.pi)
        stim1_mod1_strengths = 0.4 * ind_stim_mod1_strength / n_stim_mod1_strength + 0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4 * ind_stim_mod2_strength / n_stim_mod2_strength + 0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths

        stim1_ons = int(500 / dt)
        stim1_offs = int(1000 / dt)
        stim2_ons = int(2000 / dt)
        stim2_offs = int(2500 / dt)
        fix_offs = int(3000 / dt)
        tdim = int(3500 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_mod1_strengths = p['stim1_mod1_strengths']
        stim2_mod1_strengths = p['stim2_mod1_strengths']
        stim1_mod2_strengths = p['stim1_mod2_strengths']
        stim2_mod2_strengths = p['stim2_mod2_strengths']
        stim1_ons = int(300 / dt)
        stim1_offs = int(600 / dt)
        stim2_ons = int(p['stim_time'] / dt) + stim1_offs
        stim2_offs = int(300 / dt) + stim2_ons
        batch_size = len(stim1_locs)
        fix_offs = int(200 / dt) + stim2_offs
        tdim = int(300 / dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i] else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim1_ons),
        'stim1': (stim1_ons, stim1_offs),
        'delay1': (stim1_offs, stim2_ons),
        'stim2': (stim2_ons, stim2_offs),
        'delay2': (stim2_offs, fix_offs),
        'go1': (fix_offs, None),
    }
    return trial


def contextdelaydm1(config, mode, **kwargs):
    """Trial generator for the ``contextdelaydm1`` rule (dispatched by generate_trials)."""
    return _contextdelaydm(config, mode, 1, **kwargs)


def contextdelaydm2(config, mode, **kwargs):
    """Trial generator for the ``contextdelaydm2`` rule (dispatched by generate_trials)."""
    return _contextdelaydm(config, mode, 2, **kwargs)


def multidelaydm(config, mode, **kwargs):
    """Trial generator for the ``multidelaydm`` rule (dispatched by generate_trials)."""
    return _contextdelaydm(config, mode, 'both', **kwargs)


# ---------------------------------------------------------------------------
# Delay-match family (dmsgo / dmsnogo / dmcgo / dmcnogo)
# ---------------------------------------------------------------------------

def _dms(config, mode, matchnogo, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim1_mod = rng.choice([1, 2])
        stim2_mod = rng.choice([1, 2])
        matchs = rng.choice([0, 1], (batch_size,))
        stim_dist = rng.uniform(np.pi / 9, np.pi * 17.0 / 9.0, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist * (1 - matchs)) % (2 * np.pi)

        stim1_ons = int(rng.uniform(200, 600) / dt)
        stim1_offs = stim1_ons + int(rng.uniform(200, 1600) / dt)
        stim2_ons = stim1_offs + int(rng.uniform(200, 1600) / dt)
        tdim = stim2_ons + int(rng.uniform(300, 700) / dt)

    elif mode == 'test':
        n_stim_loc, n_mod1, n_mod2 = 40, 2, 2
        batch_size = n_stim_loc * n_mod1 * n_mod2
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size), (n_stim_loc, n_mod1, n_mod2))

        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1
        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        matchs = (1 - matchnogo) * np.ones(batch_size)
        stim2_locs = (stim1_locs + np.pi * (1 - matchs)) % (2 * np.pi)

        stim1_ons = int(500 / dt)
        stim1_offs = stim1_ons + int(500 / dt)
        stim2_ons = stim1_offs + int(1200 / dt)
        tdim = stim2_ons + int(500 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        matchs = _get_dist(stim1_locs - stim2_locs) < np.pi / 36.0
        batch_size = len(stim1_locs)
        tdim = int(2500 / dt)
        stim1_ons = int(500 / dt)
        stim1_offs = int(800 / dt)
        stim2_ons = int(2000 / dt)
        stim1_mod = 1
        stim2_mod = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = stim2_ons + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    fix_out_offs = [stim2_ons] * batch_size
    out_offs = [None] * batch_size
    for i in range(batch_size):
        if matchs[i] == matchnogo:  # If match
            fix_out_offs[i] = None  # Keep fixation
            out_offs[i] = 0  # And don't go to stimulus location

    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim1_ons),
        'stim1': (stim1_ons, stim1_offs),
        'delay1': (stim1_offs, stim2_ons),
        'go1': (stim2_ons, None),
    }
    return trial


def dmsgo(config, mode, **kwargs):
    """Trial generator for the ``dmsgo`` rule (dispatched by generate_trials)."""
    return _dms(config, mode, 0, **kwargs)


def dmsnogo(config, mode, **kwargs):
    """Trial generator for the ``dmsnogo`` rule (dispatched by generate_trials)."""
    return _dms(config, mode, 1, **kwargs)


def _dmc(config, mode, matchnogo, **kwargs):
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim1_mod = rng.choice([1, 2])
        stim2_mod = rng.choice([1, 2])
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = rng.uniform(0, 2 * np.pi, (batch_size,))

        stim1_ons = int(rng.uniform(200, 600) / dt)
        stim1_offs = stim1_ons + int(rng.uniform(200, 1600) / dt)
        stim2_ons = stim1_offs + int(rng.uniform(200, 1600) / dt)
        tdim = stim2_ons + int(rng.uniform(300, 700) / dt)

    elif mode == 'test':
        n_stim_loc, n_mod1, n_mod2 = 40, 2, 2
        batch_size = n_stim_loc * n_mod1 * n_mod2
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size), (n_stim_loc, n_mod1, n_mod2))

        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1
        n_stim_loc2 = n_stim_loc / 2
        stim1_locs_ = np.concatenate((
            (0.1 + 0.8 * np.arange(n_stim_loc2) / n_stim_loc2),
            (1.1 + 0.8 * np.arange(n_stim_loc2) / n_stim_loc2))) * np.pi
        stim1_locs = np.array([stim1_locs_[i] for i in ind_stim_loc])
        matchs = (1 - matchnogo) * np.ones(batch_size)
        stim2_locs = (stim1_locs + np.pi * (1 - matchs)) % (2 * np.pi)

        stim1_ons = int(500 / dt)
        stim1_offs = stim1_ons + int(500 / dt)
        stim2_ons = stim1_offs + int(1200 / dt)
        tdim = stim2_ons + int(500 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        batch_size = len(stim1_locs)
        tdim = int(2500 / dt)
        stim1_ons = int(500 / dt)
        stim1_offs = int(800 / dt)
        stim2_ons = int(2000 / dt)
        stim1_mod = 1
        stim2_mod = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = stim2_ons + int(100 / dt)

    stim1_cats = stim1_locs < np.pi
    stim2_cats = stim2_locs < np.pi
    matchs = stim1_cats == stim2_cats

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    fix_out_offs = [stim2_ons] * batch_size
    out_offs = [None] * batch_size
    for i in range(batch_size):
        if matchs[i] == matchnogo:  # If match
            fix_out_offs[i] = None
            out_offs[i] = 0

    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)
    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim1_ons),
        'stim1': (stim1_ons, stim1_offs),
        'delay1': (stim1_offs, stim2_ons),
        'go1': (stim2_ons, None),
    }
    return trial


def dmcgo(config, mode, **kwargs):
    """Trial generator for the ``dmcgo`` rule (dispatched by generate_trials)."""
    return _dmc(config, mode, 0, **kwargs)


def dmcnogo(config, mode, **kwargs):
    """Trial generator for the ``dmcnogo`` rule (dispatched by generate_trials)."""
    return _dmc(config, mode, 1, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

RULE_MAPPING = {
    'fdgo': fdgo,
    'reactgo': reactgo,
    'delaygo': delaygo,
    'fdanti': fdanti,
    'reactanti': reactanti,
    'delayanti': delayanti,
    'delaydm1': delaydm1,
    'delaydm2': delaydm2,
    'contextdelaydm1': contextdelaydm1,
    'contextdelaydm2': contextdelaydm2,
    'multidelaydm': multidelaydm,
    'dmsgo': dmsgo,
    'dmsnogo': dmsnogo,
    'dmcgo': dmcgo,
    'dmcnogo': dmcnogo,
}


def _build_conditions(trial, rule):
    """Build a list of per-trial condition dicts from a Trial object."""
    conditions = []
    for i in range(trial.batch_size):
        y_loc_trial = trial.y_loc[:, i]
        response_times = np.where(y_loc_trial >= 0)[0]
        response_loc = float(y_loc_trial[response_times[0]]) if len(response_times) > 0 else -1.0
        conditions.append({
            'rule': rule,
            'rule_name': RULE_NAME.get(rule, rule),
            'response_loc': response_loc,
            'epochs': {k: (v[0], v[1]) for k, v in trial.epochs.items()},
        })
    return conditions


def generate_trials(rule, n_trials=64, mode="random", sigma_x=0.01, **kwargs):
    """Generate trials for one of the Driscoll et al. (2024) flexible multitask rules.

    Args:
        rule: str, one of the 15 rule names (e.g. 'fdgo', 'delaydm1', 'dmsgo').
        n_trials: int, number of trials to generate in 'random' mode.
        mode: str, 'random', 'test', or 'psychometric'.
        sigma_x: float, input noise scale (noise std = sigma_x * sqrt(2/alpha)).
        **kwargs: passed to the task generator. 'seed' is popped and used to
            seed the random number generator; 'batch_size' is set automatically
            in 'random' mode.

    Returns:
        inputs:     (n_trials, n_t, 20) float32 numpy array.
        targets:    (n_trials, n_t, 3) float32 numpy array.
        mask:       (n_trials, n_t, 3) float32 numpy array.
        conditions: list of dicts, one per trial.
    """
    if rule not in RULE_MAPPING:
        raise ValueError("Unknown rule: {}. Supported rules: {}".format(rule, list(RULE_MAPPING.keys())))

    seed = kwargs.pop('seed', None)
    if mode == 'random':
        kwargs.setdefault('batch_size', n_trials)

    config = {
        'dt': kwargs.pop('dt', DT),
        'n_input': kwargs.pop('n_input', INPUT_DIM),
        'n_output': kwargs.pop('n_output', OUTPUT_DIM),
        'loss_type': kwargs.pop('loss_type', 'lsq'),
        'alpha': kwargs.pop('alpha', ALPHA),
        'sigma_x': sigma_x,
        'rule_start': kwargs.pop('rule_start', RULE_START),
        'rng': np.random.RandomState(seed),
    }

    trial = RULE_MAPPING[rule](config, mode, **kwargs)
    trial.add_rule(rule, on=None, off=None, strength=1.0)
    trial.add_x_noise()

    # Convert from original (time, batch, dim) to NeuralRNN (batch, time, dim).
    inputs = np.transpose(trial.x, (1, 0, 2)).astype(np.float32)
    targets = np.transpose(trial.y, (1, 0, 2)).astype(np.float32)

    # Build mask of shape (batch, time, output_dim).
    mask = trial.c_mask.reshape(trial.tdim, trial.batch_size, trial.n_output)
    mask = np.transpose(mask, (1, 0, 2)).astype(np.float32)

    conditions = _build_conditions(trial, rule)

    return inputs, targets, mask, conditions


# ---------------------------------------------------------------------------
# Unified Task-class interface (data-layer refactor; the module-level
# generate_trials above is unchanged and remains the legacy numpy shim).
# ---------------------------------------------------------------------------
from .task_base import Task  # noqa: E402


class MultitaskFlexibleTask(Task):
    """Driscoll et al. (2024) 15-task generator (unified Task interface).

    Same engine as the module-level ``generate_trials``; returns torch tensors
    and adds the unified condition keys (``n_steps`` / ``is_catch``).
    """

    name = "multitask_flexible"
    input_dim = 20
    output_dim = 3
    default_dt = 20.0
    deprecated_kwargs = {"sigma_x": "sigma_in"}
    rules = tuple(RULE_MAPPING.keys())

    def __init__(self, rule, n_trials=64, *, mode="random", sigma_in=0.01,
                 seed=None, **kwargs):
        if rule not in RULE_MAPPING:
            raise ValueError("Unknown rule: {}. Supported rules: {}".format(
                rule, list(RULE_MAPPING.keys())))
        self.rule = rule
        self.n_trials = n_trials
        self.mode = mode
        self.sigma_in = sigma_in
        self.seed = seed
        self.kwargs = kwargs

    def generate_trials(self):
        """Generate trials for the configured rule -> (inputs, targets, mask, conditions).

        conditions are per-trial dicts with the rule name plus the unified
        fields n_steps / is_catch.
        """
        import torch
        inputs, targets, mask, conditions = generate_trials(
            self.rule, n_trials=self.n_trials, mode=self.mode,
            sigma_x=self.sigma_in, seed=self.seed, **self.kwargs)
        conditions = [
            {**c, "n_steps": int(inputs.shape[1]), "is_catch": False}
            for c in conditions
        ]
        return (
            torch.as_tensor(inputs), torch.as_tensor(targets),
            torch.as_tensor(mask), conditions,
        )
