"""ECHOICE: multi-task economic choice environment (ported from Battista-2026).

Faithful gymnasium-API port of ``ECHOICE`` in
``reference_project/reinforcement_learning/Battista-2026/network_training.py``
(the ``RecordEpisodeStatisticsCustom`` logic is folded into ``info["episode"]``).

Five task rules, one sampled per trial:
    0 = standard binary choice (C vs E, certain outcomes)
    1 = risky binary choice (probabilistic outcomes)
    2 = sequential offers (working memory)
    3 = ternary choice (A vs C vs E)
    4 = bundles (C+B vs E+D)

Observation (16-dim, dt=20 ms steps):
    idx 0-4  rule one-hots (standard / risk / seq / ternary / bundle)
    idx 5    fixation flag (1 while fixation must be held)
    idx 6-10 good quantities / 10 (n_a, n_b, n_c, n_d, n_e)
    idx 11-15 offer probabilities (p_a .. p_e)
    baseline u0 = 0.2 plus observation noise (prefactor sqrt(2/alpha) * sigma)

Actions (Discrete(4)): 0 = hold fixation / no answer, 1 = choose A,
2 = choose C (or bundle C+B), 3 = choose E (or E+D).

Rewards: breaking fixation early -> -1 and trial abort; never deciding ->
-1 at trial end; valid choice -> realized quantity * value (drawn with the
offer probability); good values vE = 1 with conversions A_to_E = 3, B_to_E = 2.5,
C_to_E = 2, D_to_E = 1.5.

Deviations from the reference implementation (behavior-identical):
    - gymnasium API (reset -> (obs, info); step -> 5-tuple) instead of old gym;
    - per-env ``numpy.random.Generator`` instead of global ``np.random``;
    - episode statistics are computed inside the env (info["episode"]) instead of
      a wrapper; ``evt``/expected values are included for analysis.
"""
from __future__ import annotations

import numpy as np

try:  # spaces: prefer gymnasium, fall back to gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    from gym import spaces

RULE_NAMES = {0: "standard", 1: "risk", 2: "seq", 3: "ternary", 4: "bundle"}


class EchoiceEnv:
    """Multi-task economic choice environment. See module docstring."""

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,))
    action_space = spaces.Discrete(4)

    def __init__(self, *, dt: float = 20., tau: float = 100.,
                 t_fix: float = 1000., t_dec: float = 1000., t_wait: float = 1000.,
                 t_deli: float = 1000., t_delf: float = 1500., t_offer: float = 1000.,
                 sigma: float = 0.01, u0: float = 0.2,
                 A_to_E: float = 3., B_to_E: float = 2.5, C_to_E: float = 2.,
                 D_to_E: float = 1.5, vE: float = 1.,
                 rules: tuple[int, ...] = (0, 1, 2, 3, 4), seed: int | None = None):
        self.dt = dt
        self.Nu = 16
        self.Na = 4
        self.alpha = dt / tau
        self.t_fix = t_fix
        self.t_dec = t_dec
        self.t_wait = t_wait
        self.t_deli = t_deli
        self.t_delf = t_delf
        self.t_offer = t_offer
        self.sigma = sigma
        self.u0 = u0
        self.A_to_E = A_to_E
        self.B_to_E = B_to_E
        self.C_to_E = C_to_E
        self.D_to_E = D_to_E
        self.vE = vE
        self.prefactor = np.sqrt(2 / self.alpha) * sigma
        self.rules = tuple(rules)
        if not self.rules or any(r not in RULE_NAMES for r in self.rules):
            raise ValueError(f"rules must be a non-empty subset of {tuple(RULE_NAMES)}")
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options=None):
        """Start a new trial. Returns (obs, info); obs is all zeros (as in the reference)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        j = int(500. / self.dt)
        self.u_next = np.zeros(self.Nu)
        self.t_fixn = self.t_fix + rng.integers(-j, j + 1) * self.dt
        self.t_rulen = self.t_fix + rng.integers(-j, j + 1) * self.dt
        self.t_decn = self.t_dec + rng.integers(-j, j + 1) * self.dt
        self.t_offer1n = self.t_offer + rng.integers(-j, j + 1) * self.dt
        self.t_offer2n = self.t_offer + rng.integers(-j, j + 1) * self.dt
        self.t_wait1n = self.t_wait + rng.integers(-j, j + 1) * self.dt
        self.t_wait2n = self.t_wait + rng.integers(-j, j + 1) * self.dt
        self.t_stimn = rng.integers(int(500. / self.dt), int(1500. / self.dt) + 1) * self.dt
        self.t_deln = 0.

        self.t = 0.
        self.scratch = 0.
        self._ep_return = 0.
        self._evt = -1.

        self.ne_max = 10.
        self.nd_max = 10. / self.D_to_E
        self.nc_max = 10. / self.C_to_E
        self.nb_max = 10. / self.B_to_E
        self.na_max = 10. / self.A_to_E

        self.rule = self.rules[rng.integers(0, len(self.rules))]
        self._sample_trial()
        return self.u_next.copy(), {"rule": self.rule}

    def step(self, action: int):
        """Advance one dt. Returns (obs, reward, terminated, truncated, info)."""
        rng = self._rng
        evt = -1.  # expected value of the chosen offer (-1: no valid choice)
        reward = 0.

        # --- reward for the action taken at the current time ---
        if self.t < self.t_gon:
            if action != 0:
                reward += -1.
                self.scratch = 1.
        else:
            if action != 0:
                self.scratch = 1.
                if self.rule == 0:
                    if action == 2:
                        reward += self.rc
                        evt = self.evc
                    if action == 3:
                        reward += self.re
                        evt = self.eve
                    if action == 1:
                        reward += -1.
                        evt = -1.
                if self.rule == 1:
                    if action == 2:
                        reward += self.rc * rng.choice([0., 1.], p=[1 - self.pro_c, self.pro_c])
                        evt = self.evc
                    if action == 3:
                        reward += self.re * rng.choice([0., 1.], p=[1 - self.pro_e, self.pro_e])
                        evt = self.eve
                    if action == 1:
                        reward += -1.
                        evt = -1.
                if self.rule == 3:
                    if action == 1:
                        reward += self.ra * rng.choice([0., 1.], p=[1 - self.pro_a, self.pro_a])
                        evt = self.eva
                    if action == 2:
                        reward += self.rc * rng.choice([0., 1.], p=[1 - self.pro_c, self.pro_c])
                        evt = self.evc
                    if action == 3:
                        reward += self.re * rng.choice([0., 1.], p=[1 - self.pro_e, self.pro_e])
                        evt = self.eve
                if self.rule == 4:
                    if action == 1:
                        reward += -1.
                        evt = -1.
                    if action == 2:
                        reward += (self.rc * rng.choice([0., 1.], p=[1 - self.pro_c, self.pro_c])
                                   + self.rb * rng.choice([0., 1.], p=[1 - self.pro_b, self.pro_b]))
                        evt = self.evc + self.evb
                    if action == 3:
                        reward += (self.re * rng.choice([0., 1.], p=[1 - self.pro_e, self.pro_e])
                                   + self.rd * rng.choice([0., 1.], p=[1 - self.pro_d, self.pro_d]))
                        evt = self.eve + self.evd
                if self.rule == 2:
                    if action == 2:
                        reward += self.rc * rng.choice([0., 1.], p=[1 - self.pro_c, self.pro_c])
                        evt = self.evc
                    if action == 3:
                        reward += self.re * rng.choice([0., 1.], p=[1 - self.pro_e, self.pro_e])
                        evt = self.eve
                    if action == 1:
                        reward += -1.
                        evt = -1.
        self._evt = evt

        # --- generate the next observation ---
        noise = self.prefactor * rng.normal(0., 1., self.Nu)
        self.u_next = noise + self.u0
        if self.t < self.t_gon:
            self.u_next[5] += 1.  # fixation flag

        stim_on = (self.t >= self.t_fixn + self.t_rulen) and (self.t < self.t_gon)
        if self.rule == 0:
            if self.t >= self.t_fixn:
                self.u_next[0] += 1.
            if stim_on:
                self.u_next[8] += self.nc / 10.
                self.u_next[10] += self.ne / 10.
                self.u_next[13] += 1.
                self.u_next[15] += 1.
        if self.rule == 1:
            if self.t >= self.t_fixn:
                self.u_next[1] += 1.
            if stim_on:
                self.u_next[8] += self.nc / 10.
                self.u_next[10] += self.ne / 10.
                self.u_next[13] += self.pro_c
                self.u_next[15] += self.pro_e
        if self.rule == 3:
            if self.t >= self.t_fixn:
                self.u_next[3] += 1.
            if stim_on:
                self.u_next[6] += self.na / 10.
                self.u_next[8] += self.nc / 10.
                self.u_next[10] += self.ne / 10.
                self.u_next[11] += self.pro_a
                self.u_next[13] += self.pro_c
                self.u_next[15] += self.pro_e
        if self.rule == 4:
            if self.t >= self.t_fixn:
                self.u_next[4] += 1.
            if stim_on:
                self.u_next[7] += self.nb / 10.
                self.u_next[8] += self.nc / 10.
                self.u_next[9] += self.nd / 10.
                self.u_next[10] += self.ne / 10.
                self.u_next[12] += self.pro_b
                self.u_next[13] += self.pro_c
                self.u_next[14] += self.pro_d
                self.u_next[15] += self.pro_e
        if self.rule == 2:
            if self.t >= self.t_fixn:
                self.u_next[2] += 1.
            offer1_on = (self.t >= self.t_fixn + self.t_rulen
                         and self.t < self.t_fixn + self.t_rulen + self.t_offer1n)
            offer2_on = (self.t >= self.t_fixn + self.t_rulen + self.t_offer1n + self.t_wait1n
                         and self.t < self.t_fixn + self.t_rulen + self.t_offer1n
                         + self.t_wait1n + self.t_offer2n)
            if self.c_first == 1:
                if offer1_on:
                    self.u_next[8] += self.nc / 10.
                    self.u_next[13] += self.pro_c
                if offer2_on:
                    self.u_next[10] += self.ne / 10.
                    self.u_next[15] += self.pro_e
            else:
                if offer1_on:
                    self.u_next[10] += self.ne / 10.
                    self.u_next[15] += self.pro_e
                if offer2_on:
                    self.u_next[8] += self.nc / 10.
                    self.u_next[13] += self.pro_c

        self.u_next = np.maximum(self.u_next, 0)

        if self.t >= self.t_trial_maxn:
            self.scratch = 1.
            reward += -1.

        self.t += self.dt
        self._ep_return += reward

        terminated = self.scratch != 0.
        info = {}
        if terminated:
            info["episode"] = self._episode_info(action)
        return self.u_next.copy(), reward, terminated, False, info

    # ------------------------------------------------------------- internals
    def _jittered(self, base: float) -> float:
        j = int(500. / self.dt)
        return base + self._rng.integers(-j, j + 1) * self.dt

    def _draw_offer(self, n_max: float, certain: bool):
        """Draw (quantity, probability) with the reference redraw rule n*p >= n_max/10."""
        expq = 0.
        while expq < n_max / 10.:
            n = self._rng.uniform(0, 10) * (n_max / 10.)
            p = 1. if certain else self._rng.uniform(0, 1)
            expq = n * p
        return n, p

    def _sample_trial(self) -> None:
        """Sample offers / timing for the current rule (called by reset)."""
        sim_offers = self.t_fixn + self.t_rulen + self.t_stimn
        if self.rule in (0, 1, 3, 4):
            self.t_gon = sim_offers
            self.t_trial_maxn = sim_offers + self.t_decn
            self.c_first = 2  # simultaneous offers
        else:  # rule == 2, sequential offers
            self.t_gon = (self.t_fixn + self.t_rulen + self.t_offer1n + self.t_wait1n
                          + self.t_offer2n + self.t_wait2n + self.t_deln)
            self.t_trial_maxn = self.t_gon + self.t_decn
            self.c_first = int(self._rng.integers(0, 2))  # 1: C first, 0: E first
        self.N_t_max = int(self.t_trial_maxn / self.dt)

        certain = self.rule == 0
        # defaults: absent offers
        self.na = self.nb = self.nd = 0.
        self.pro_a = self.pro_b = self.pro_d = 0.
        if self.rule in (0, 1, 2):
            self.nc, self.pro_c = self._draw_offer(self.nc_max, certain)
            self.ne, self.pro_e = self._draw_offer(self.ne_max, certain)
        elif self.rule == 3:
            self.na, self.pro_a = self._draw_offer(self.na_max, False)
            self.nc, self.pro_c = self._draw_offer(self.nc_max, False)
            self.ne, self.pro_e = self._draw_offer(self.ne_max, False)
        else:  # rule == 4
            self.nb, self.pro_b = self._draw_offer(self.nb_max, False)
            self.nc, self.pro_c = self._draw_offer(self.nc_max, False)
            self.nd, self.pro_d = self._draw_offer(self.nd_max, False)
            self.ne, self.pro_e = self._draw_offer(self.ne_max, False)

        self.re = self.ne * self.vE / 10.
        self.rd = self.nd * self.D_to_E * self.vE / 10.
        self.rc = self.nc * self.C_to_E * self.vE / 10.
        self.rb = self.nb * self.B_to_E * self.vE / 10.
        self.ra = self.na * self.A_to_E * self.vE / 10.
        self.eve = self.re * self.pro_e
        self.evd = self.rd * self.pro_d
        self.evc = self.rc * self.pro_c
        self.evb = self.rb * self.pro_b
        self.eva = self.ra * self.pro_a

    def _episode_info(self, action: int) -> dict:
        """Per-trial statistics (port of RecordEpisodeStatisticsCustom)."""
        tempo = self.t
        rt = tempo - self.t_gon
        n_dec = 1. if (tempo >= self.t_gon and self._evt >= 0.) else 0.
        n_corr = 0.
        if n_dec > 0.5:
            if self.rule in (0, 1, 2):
                best = max(self.evc, self.eve)
            elif self.rule == 3:
                best = max(self.eva, self.evc, self.eve)
            else:
                best = max(self.evc + self.evb, self.eve + self.evd)
            n_corr = 1. if abs(self._evt - best) < 0.05 else 0.
        return {
            "r": self._ep_return, "ret": self._ep_return, "t": tempo,
            "n_corr": n_corr, "n_dec": n_dec, "t_go": self.t_gon, "RT": rt,
            "ra": self.ra, "rb": self.rb, "rc": self.rc, "rd": self.rd, "re": self.re,
            "pa": self.pro_a, "pb": self.pro_b, "pc": self.pro_c,
            "pd": self.pro_d, "pe": self.pro_e,
            "na": self.na, "nb": self.nb, "nc": self.nc, "nd": self.nd, "ne": self.ne,
            "eva": self.eva, "evb": self.evb, "evc": self.evc,
            "evd": self.evd, "eve": self.eve, "evt": self._evt,
            "c_first": self.c_first, "rule": self.rule, "choice": action,
            "t_fix": self.t_fixn, "t_rule": self.t_rulen, "t_dec": self.t_decn,
            "t_wait1": self.t_wait1n, "t_wait2": self.t_wait2n,
            "t_offer1": self.t_offer1n, "t_offer2": self.t_offer2n,
            "t_stim": self.t_stimn, "t_del": self.t_deln,
        }
