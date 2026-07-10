"""Yang et al. (2019) 20-task multitask dataset.

Also referred to as the "multitask" or "task-set" dataset from the
prefrontal-cortex multitask RNN literature. The dataset contains 20
cognitive tasks that share a common input/output ring architecture and are
distinguished by a one-hot rule input.

Paper reference:
    Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T., & Wang, X. J.
    (2019). Task representations in neural networks trained to perform many
    cognitive tasks. Nature Neuroscience, 22(2), 297-306.

Input format:
    inputs: (n_trials, n_t, 85) float32 numpy array
        - Channel 0: fixation input.
        - Channels 1-32: stimulus ring 1 (32-unit Gaussian population code).
        - Channels 33-64: stimulus ring 2 (32-unit Gaussian population code).
        - Channels 65-84: one-hot rule input (20 tasks).

Target format:
    targets: (n_trials, n_t, 33) float32 numpy array
        - Channel 0: fixation target.
        - Channels 1-32: output ring (32-unit Gaussian / one-hot population code).

Mask format:
    mask: (n_trials, n_t, 33) float32 numpy array
        Per-time-step weight expanded across outputs. Pre-response period is
        weighted 1, post-response period is weighted 5, and the mask is
        normalized to mean 1 (matching the original softmax-loss weighting).

Task list (20 tasks):
    Go family:
        fdgo      : Go (stimulus shown from fixation onset).
        reactgo   : Reaction-time Go (stimulus shown after fixation offset).
        delaygo   : Delayed Go (stimulus shown, removed, then go cue).
    Anti-go family:
        fdanti    : Anti-Go (saccade opposite to stimulus).
        reactanti : Reaction-time Anti-Go.
        delayanti : Delayed Anti-Go.
    Decision-making family:
        dm1       : Single-modality decision making (ring 1).
        dm2       : Single-modality decision making (ring 2).
        delaydm1  : Delayed single-modality decision making (ring 1).
        delaydm2  : Delayed single-modality decision making (ring 2).
    Context decision-making family:
        contextdm1       : Context-dependent DM, attend to ring 1.
        contextdm2       : Context-dependent DM, attend to ring 2.
        contextdelaydm1  : Context-dependent delayed DM, attend to ring 1.
        contextdelaydm2  : Context-dependent delayed DM, attend to ring 2.
    Multisensory decision-making family:
        multidm       : Multisensory DM (integrate both rings).
        multidelaydm  : Multisensory delayed DM (integrate both rings).
    Delayed-match family:
        dmsgo   : Delayed match-to-sample, Go on match.
        dmsnogo : Delayed match-to-sample, No-Go on match.
        dmcgo   : Delayed match-to-category, Go on category match.
        dmcnogo : Delayed match-to-category, No-Go on category match.

Notes:
    - This module ports the trial-generation logic from
      reference_project/multitask/multitask/task.py.
    - Internally, trials are built with the original (time, batch, dim) layout;
      arrays are transposed to NeuralRNN's (batch, time, dim) layout at return.
    - Timing values are in ms and converted to time steps with dt=20 ms.
    - Input noise is sigma_x * sqrt(2 / alpha) with alpha=0.2, matching the
      original continuous-time Ornstein-Uhlenbeck formulation.
"""

from __future__ import division
import numpy as np


# ---------------------------------------------------------------------------
# Rule bookkeeping
# ---------------------------------------------------------------------------

RULES_ALL = [
    'fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
    'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
    'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
    'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo',
]

RULE_INDEX_MAP = {rule: idx for idx, rule in enumerate(RULES_ALL)}

RULE_NAME = {
    'reactgo': 'RT Go',
    'delaygo': 'Dly Go',
    'fdgo': 'Go',
    'dm1': 'DM 1',
    'dm2': 'DM 2',
    'contextdm1': 'Ctx DM 1',
    'contextdm2': 'Ctx DM 2',
    'multidm': 'MultSen DM',
    'delaydm1': 'Dly DM 1',
    'delaydm2': 'Dly DM 2',
    'contextdelaydm1': 'Ctx Dly DM 1',
    'contextdelaydm2': 'Ctx Dly DM 2',
    'multidelaydm': 'MultSen Dly DM',
    'reactanti': 'RT Anti',
    'delayanti': 'Dly Anti',
    'fdanti': 'Anti',
    'dmsgo': 'DMS',
    'dmsnogo': 'DNMS',
    'dmcgo': 'DMC',
    'dmcnogo': 'DNMC',
}


def _get_dist(original_dist):
    """Distance on a ring with periodic boundary conditions."""
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


# ---------------------------------------------------------------------------
# Trial helper class
# ---------------------------------------------------------------------------

class Trial(object):
    """A batch of trials, built with the original (time, batch, dim) layout."""

    def __init__(self, config, tdim, batch_size):
        self.float_type = 'float32'
        self.config = config
        self.dt = self.config['dt']

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_eachring)

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:, :, :] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation,
        # (0, 2*pi) for response.
        self.y_loc = -np.ones((tdim, batch_size), dtype=self.float_type)

        self._sigma_x = config['sigma_x'] * np.sqrt(2 / config['alpha'])

    def expand(self, var):
        """Expand an int/float to a list of length batch_size."""
        if var is None:
            return [None] * self.batch_size
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or target output to the batch.

        Args:
            loc_type: 'fix_in', 'stim', 'fix_out', or 'out'.
            locs: array/list of locations, used for stim/out.
            ons: int or list, onset time step (None -> 0).
            offs: int or list, offset time step (None -> tdim).
            strengths: float or list, strength multiplier.
            mods: int or list, modality index (1-indexed).
        """
        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]:offs[i], i, 0] = 1
            elif loc_type == 'stim':
                # mods[i] is 1-indexed
                start = 1 + (mods[i] - 1) * self.n_eachring
                end = 1 + mods[i] * self.n_eachring
                self.x[ons[i]:offs[i], i, start:end] += self.add_x_loc(locs[i]) * strengths[i]
            elif loc_type == 'fix_out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]:offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i], i, 1:] += self.add_y_loc(locs[i]) * strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]:offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]:offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type: {}'.format(loc_type))

    def add_x_noise(self):
        """Add input noise to the trial inputs."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        """Build the per-time-step cost mask.

        Pre-response period is weighted 1; post-response period is weighted 5.
        For the softmax loss formulation the mask is normalized to mean 1.
        """
        pre_on = int(100 / self.dt)  # never check the first 100 ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[post_ons[i]:, i, :] = 5.0
                c_mask[pre_on:pre_offs[i], i, :] = 1.0
            c_mask[:, :, 0] *= 2.0  # Fixation is important
            self.c_mask = c_mask.reshape((self.tdim * self.batch_size, self.n_output))
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                c_mask[post_ons[i]:, i] = 5.0
                c_mask[pre_on:pre_offs[i], i] = 1.0
            self.c_mask = c_mask.reshape((self.tdim * self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_rule(self, rule, on=None, off=None, strength=1.0):
        """Add a one-hot rule input to the rule channels."""
        if isinstance(rule, int):
            ind_rule = self.config['rule_start'] + rule
        else:
            ind_rule = self.config['rule_start'] + RULE_INDEX_MAP[rule]
        self.x[on:off, :, ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Input population activity for a stimulus location."""
        dist = _get_dist(x_loc - self.pref)
        dist /= np.pi / 8
        return 0.8 * np.exp(-dist ** 2 / 2)

    def add_y_loc(self, y_loc):
        """Target population activity for a response location."""
        dist = _get_dist(y_loc - self.pref)
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi / 8
            y = 0.8 * np.exp(-dist ** 2 / 2)
        else:
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.0
        return y


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def _delaygo_core(config, mode, anti_response, **kwargs):
    """Core logic for delaygo / delayanti."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_ons = int(rng.choice([300, 500, 700]) / dt)
        stim_offs = stim_ons + int(rng.choice([200, 400, 600]) / dt)
        fix_offs = stim_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
        tdim = fix_offs + int(500 / dt)
        stim_mod = rng.choice([1, 2])
    elif mode == 'test':
        tdim = int(2500 / dt)
        n_stim_loc, n_stim_mod = 20, 2
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
    return _delaygo_core(config, mode, False, **kwargs)


def delayanti(config, mode, **kwargs):
    return _delaygo_core(config, mode, True, **kwargs)


def _reactgo_core(config, mode, anti_response, **kwargs):
    """Core logic for reactgo / reactanti."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_ons = int(rng.uniform(500, 2500) / dt)
        tdim = int(500 / dt) + stim_ons
        stim_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim_mod = rng.choice([1, 2])
    elif mode == 'test':
        tdim = int(2500 / dt)
        n_stim_loc, n_stim_mod = 20, 2
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
    return _reactgo_core(config, mode, False, **kwargs)


def reactanti(config, mode, **kwargs):
    return _reactgo_core(config, mode, True, **kwargs)


def _fdgo_core(config, mode, anti_response, **kwargs):
    """Core logic for fdgo / fdanti."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_locs = rng.rand(batch_size) * 2 * np.pi
        stim_mod = rng.choice([1, 2])
        stim_ons = int(rng.uniform(300, 700) / dt)
        fix_offs = stim_ons + int(rng.uniform(500, 1500) / dt)
        tdim = int(500 / dt) + fix_offs
    elif mode == 'test':
        tdim = int(2000 / dt)
        n_stim_loc, n_stim_mod = 20, 2
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
    return _fdgo_core(config, mode, False, **kwargs)


def fdanti(config, mode, **kwargs):
    return _fdgo_core(config, mode, True, **kwargs)


def _contextdm_genstim(batch_size, rng, stim_coh_range=None):
    """Generate correlated stimulus strengths for context DM tasks."""
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.array([0.16, 0.32, 0.64])
    stim_coh = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([+1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh * stim_sign
    stim2_strengths = stim_mean - stim_coh * stim_sign
    return stim1_strengths, stim2_strengths


def _contextdm(config, mode, attend_mod, **kwargs):
    """Context-dependent decision making (contextdm1, contextdm2, multidm)."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']

        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if config.get('easy_task', False):
            stim_coh_range *= 10

        if attend_mod in (1, 2):
            stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        else:
            stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod12_diff = stim1_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2
            stim2_mod12_diff = stim2_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

        stim_on = int(rng.uniform(100, 400) / dt)
        stim_ons = (np.ones(batch_size) * stim_on).astype(int)
        stim_dur = int(rng.choice([400, 800, 1600]) / dt)
        stim_offs = stim_ons + stim_dur
        delay_dur = 0
        fix_offs = stim_offs + delay_dur
        tdim = stim_on + stim_dur + delay_dur + int(500 / dt)

    elif mode == 'test':
        tdim = int(2000 / dt)
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = 20, 5, 5
        batch_size = n_stim_loc * n_stim_mod1_strength * n_stim_mod2_strength
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(
            range(batch_size), (n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength))
        fix_offs = int(1500 / dt)
        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim2_locs = (stim1_locs + np.pi) % (2 * np.pi)
        stim1_mod1_strengths = 0.4 * ind_stim_mod1_strength / n_stim_mod1_strength + 0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4 * ind_stim_mod2_strength / n_stim_mod2_strength + 0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths
        stim_ons = int(500 / dt)
        stim_offs = int(1500 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_mod1_strengths = p['stim1_mod1_strengths']
        stim2_mod1_strengths = p['stim2_mod1_strengths']
        stim1_mod2_strengths = p['stim1_mod2_strengths']
        stim2_mod2_strengths = p['stim2_mod2_strengths']
        stim_time = int(p['stim_time'] / dt)
        batch_size = len(stim1_locs)
        stim_ons = int(500 / dt)
        stim_offs = stim_ons + stim_time
        fix_offs = stim_offs
        tdim = int(500 / dt) + fix_offs

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
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i] else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, stim_offs),
        'go1': (fix_offs, None),
    }
    return trial


def contextdm1(config, mode, **kwargs):
    return _contextdm(config, mode, 1, **kwargs)


def contextdm2(config, mode, **kwargs):
    return _contextdm(config, mode, 2, **kwargs)


def multidm(config, mode, **kwargs):
    return _contextdm(config, mode, 'both', **kwargs)


def _dm(config, mode, stim_mod, **kwargs):
    """Single-modality decision making (dm1, dm2)."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if config.get('easy_task', False):
            stim_coh_range *= 10

        stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
        stims_coh = rng.choice(stim_coh_range, (batch_size,))
        stims_sign = rng.choice([1, -1], (batch_size,))
        stim1_strengths = stims_mean + stims_coh * stims_sign
        stim2_strengths = stims_mean - stims_coh * stims_sign

        stim_on = int(rng.uniform(100, 400) / dt)
        stim_ons = (np.ones(batch_size) * stim_on).astype(int)
        stim_dur = int(rng.choice([400, 800, 1600]) / dt)
        fix_offs = (stim_ons + stim_dur).astype(int)
        tdim = stim_on + stim_dur + int(500 / dt)

    elif mode == 'test':
        tdim = int(2500 / dt)
        n_stim_loc, n_stim1_strength = 20, 5
        batch_size = n_stim_loc * n_stim1_strength
        ind_stim_loc, ind_stim1_strength = np.unravel_index(range(batch_size), (n_stim_loc, n_stim1_strength))
        fix_offs = int(2000 / dt)
        stim1_locs = 2 * np.pi * ind_stim_loc / n_stim_loc
        stim2_locs = (stim1_locs + np.pi) % (2 * np.pi)
        stim1_strengths = 0.4 * ind_stim1_strength / n_stim1_strength + 0.8
        stim2_strengths = 2 - stim1_strengths
        stim_ons = int(500 / dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_strengths = p['stim1_strengths']
        stim2_strengths = p['stim2_strengths']
        stim_time = int(p['stim_time'] / dt)
        batch_size = len(stim1_locs)
        stim_ons = int(300 / dt)
        fix_offs = int(300 / dt) + stim_time
        tdim = int(400 / dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons = fix_offs + int(100 / dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if stim1_strengths[i] > stim2_strengths[i] else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {
        'fix1': (None, stim_ons),
        'stim1': (stim_ons, fix_offs),
        'go1': (fix_offs, None),
    }
    return trial


def dm1(config, mode, **kwargs):
    return _dm(config, mode, 1, **kwargs)


def dm2(config, mode, **kwargs):
    return _dm(config, mode, 2, **kwargs)


def _delaydm(config, mode, stim_mod, **kwargs):
    """Delayed single-modality decision making (delaydm1, delaydm2)."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stims_mean = rng.uniform(0.8, 1.2, (batch_size,))
        stim_coh_range = np.array([0.08, 0.16, 0.32])
        if config.get('easy_task', False):
            stim_coh_range *= 2

        stims_coh = rng.choice(stim_coh_range, (batch_size,))
        stims_sign = rng.choice([1, -1], (batch_size,))
        stim1_strengths = stims_mean + stims_coh * stims_sign
        stim2_strengths = stims_mean - stims_coh * stims_sign

        stim1_ons = int(rng.choice([200, 400, 600]) / dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
        stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
        stim2_offs = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
        fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        tdim = int(3000 / dt)
        n_stim_loc, n_stim1_strength = 20, 5
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
    return _delaydm(config, mode, 1, **kwargs)


def delaydm2(config, mode, **kwargs):
    return _delaydm(config, mode, 2, **kwargs)


def _contextdelaydm(config, mode, attend_mod, **kwargs):
    """Context-dependent delayed decision making."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim_dist = rng.uniform(0.5 * np.pi, 1.5 * np.pi, (batch_size,)) * rng.choice([-1, 1], (batch_size,))
        stim1_locs = rng.uniform(0, 2 * np.pi, (batch_size,))
        stim2_locs = (stim1_locs + stim_dist) % (2 * np.pi)

        stim_coh_range = np.array([0.08, 0.16, 0.32])
        if config.get('easy_task', False):
            stim_coh_range *= 2

        if attend_mod in (1, 2):
            stim1_mod1_strengths, stim2_mod1_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
        else:
            stim1_strengths, stim2_strengths = _contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod12_diff = stim1_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff / 2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff / 2
            stim2_mod12_diff = stim2_strengths * rng.uniform(0.2, 0.8, (batch_size,)) * rng.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff / 2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff / 2

        stim1_ons = int(rng.choice([200, 400, 600]) / dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
        stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
        stim2_offs = stim2_ons + int(rng.choice([200, 400, 600]) / dt)
        fix_offs = stim2_offs + int(rng.uniform(100, 300) / dt)
        tdim = fix_offs + int(500 / dt)

    elif mode == 'test':
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = 20, 5, 5
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
    return _contextdelaydm(config, mode, 1, **kwargs)


def contextdelaydm2(config, mode, **kwargs):
    return _contextdelaydm(config, mode, 2, **kwargs)


def multidelaydm(config, mode, **kwargs):
    return _contextdelaydm(config, mode, 'both', **kwargs)


def _dms(config, mode, matchnogo, **kwargs):
    """Delayed match-to-sample (dmsgo / dmsnogo)."""
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

        stim1_ons = int(rng.choice([200, 400, 600]) / dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
        stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
        tdim = stim2_ons + int(500 / dt)

    elif mode == 'test':
        n_stim_loc, n_mod1, n_mod2 = 20, 2, 2
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
    return _dms(config, mode, 0, **kwargs)


def dmsnogo(config, mode, **kwargs):
    return _dms(config, mode, 1, **kwargs)


def _dmc(config, mode, matchnogo, **kwargs):
    """Delayed match-to-category (dmcgo / dmcnogo)."""
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        stim1_mod = rng.choice([1, 2])
        stim2_mod = rng.choice([1, 2])
        stim_locs_pool = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * np.pi
        stim1_locs = rng.choice(stim_locs_pool, size=(batch_size,))
        stim2_locs = rng.choice(stim_locs_pool, size=(batch_size,))

        stim1_ons = int(rng.choice([200, 400, 600]) / dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600]) / dt)
        stim2_ons = stim1_offs + int(rng.choice([200, 400, 800, 1600]) / dt)
        tdim = stim2_ons + int(rng.choice([200, 400, 600]) / dt)

    elif mode == 'test':
        n_stim_loc, n_mod1, n_mod2 = 20, 2, 2
        batch_size = n_stim_loc * n_mod1 * n_mod2
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size), (n_stim_loc, n_mod1, n_mod2))
        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1
        n_stim_loc2 = n_stim_loc // 2
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
    return _dmc(config, mode, 0, **kwargs)


def dmcnogo(config, mode, **kwargs):
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
    'dm1': dm1,
    'dm2': dm2,
    'contextdm1': contextdm1,
    'contextdm2': contextdm2,
    'multidm': multidm,
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
    """Generate trials for one of the Yang et al. (2019) multitask rules.

    Args:
        rule: str, one of the 20 rule names (e.g. 'fdgo', 'dm1', 'dmsgo').
        n_trials: int, number of trials to generate in 'random' mode.
        mode: str, 'random', 'test', or 'psychometric'.
        sigma_x: float, input noise scale (noise std = sigma_x * sqrt(2/alpha)).
        **kwargs: passed to the task generator. 'seed' is popped and used to
            seed the random number generator; 'batch_size' is set automatically
            in 'random' mode.

    Returns:
        inputs:     (n_trials, n_t, 85) float32 numpy array.
        targets:    (n_trials, n_t, 33) float32 numpy array.
        mask:       (n_trials, n_t, 33) float32 numpy array.
        conditions: list of dicts, one per trial.
    """
    if rule not in RULE_MAPPING:
        raise ValueError("Unknown rule: {}. Supported rules: {}".format(rule, list(RULE_MAPPING.keys())))

    seed = kwargs.pop('seed', None)
    if mode == 'random':
        kwargs.setdefault('batch_size', n_trials)

    config = {
        'dt': kwargs.pop('dt', 20),
        'n_eachring': kwargs.pop('n_eachring', 32),
        'n_input': kwargs.pop('n_input', 85),
        'n_output': kwargs.pop('n_output', 33),
        'loss_type': kwargs.pop('loss_type', 'lsq'),
        'alpha': kwargs.pop('alpha', 0.2),
        'sigma_x': sigma_x,
        'rule_start': kwargs.pop('rule_start', 65),
        'rng': np.random.RandomState(seed),
    }

    trial = RULE_MAPPING[rule](config, mode, **kwargs)
    trial.add_rule(rule, on=None, off=None, strength=1.0)
    trial.add_x_noise()

    # Convert from original (time, batch, dim) to NeuralRNN (batch, time, dim).
    inputs = np.transpose(trial.x, (1, 0, 2)).astype(np.float32)
    targets = np.transpose(trial.y, (1, 0, 2)).astype(np.float32)

    # Build mask of shape (batch, time, output_dim).
    if config['loss_type'] == 'lsq':
        # c_mask is already (time*batch, output_dim); reshape and transpose.
        mask = trial.c_mask.reshape(trial.tdim, trial.batch_size, trial.n_output)
        mask = np.transpose(mask, (1, 0, 2)).astype(np.float32)
    else:
        # c_mask is (time*batch,); expand to output dim and transpose.
        mask_2d = trial.c_mask.reshape(trial.tdim, trial.batch_size)
        mask = np.tile(mask_2d[:, :, None], (1, 1, trial.n_output))
        mask = np.transpose(mask, (1, 0, 2)).astype(np.float32)

    conditions = _build_conditions(trial, rule)

    return inputs, targets, mask, conditions
