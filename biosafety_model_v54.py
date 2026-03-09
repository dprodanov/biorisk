# -*- coding: utf-8 -*-
"""

@author: Dimiter Prodanov
IICT-BAS, BioPrecis

Biosafety Log-Risk Markov Framework
====================================
Implementation of the methodology described in:
"Making Biorisk Measurable: A Bayesian Framework for Laboratory Risk Management"

Version: 5.4  
date 06 March 2026 - natech handling
4.0 - cause-class tagging
date 23 Feb 2026

"""
import numpy as np
from scipy import stats
import pandas as pd
from typing import Optional

version="5.4"

class BiosafetyModel:
    """
    5-state absorbing Markov chain biosafety model — v5.4

    States: S0=Normal, S1=Minor, S2=Serious, S3=Critical, S4=Disaster (absorbing)

    Three-metric ROI framework:
      L       : per-procedure log-risk from S0  (training metric)
      logM_S1 : log10(MFPT from S1)             (maintenance metric)
      logM_S2 : log10(MFPT from S2)             (inspection metric)

    Edge classes:
      H   : (0,1),(0,4),(1,0)           — training only
      HE  : (1,4)                       — training + maintenance (dual)
      E   : (1,2),(2,3),(2,0)           — maintenance only
      EO  : (2,4),(3,4)                 — maintenance + inspection (dual)
      -   : (2,1),(3,2)                 — uncontrolled (spontaneous de-escalation)

    Intervention effects (log10 units added to base rate):
      train_effect  = 2.0 * (1 - 1/(1 + (training/50)^2.5))
      maint_effect  = freq_effect * adh_mult * gap_penalty
      insp_effect   = 1.0 if inspection >= 70 else 0.0
    """

    STATE_NAMES = ["S0_Normal", "S1_Minor", "S2_Serious", "S3_Critical", "S4_Disaster"]

    # BASE_RATES: log10 barrier heights; P[i,j] ~ 10^{-base} before interventions.
    # Higher value = lower raw probability. Interventions add positive increments.
    #
    #  Escalation edges (->higher severity): barriers 2.5–6.0
    #  De-escalation edges (recovery):       barriers 1.0–2.0
    #  Direct-to-disaster shortcuts:         barriers 4.0–6.0 (rare by design)
    #  (3,4) = 2.0: lowest escalation barrier — critical state is proximate to disaster
    # Natech lookup: exogenous S0->S4 rate (not in BASE_RATES)
    # b_04 = -log10(P_natech_per_procedure); larger b = safer/remote site
    NATECH_PROFILES = {
        "inland_low_seismic": 7.5,   # Brussels, Prague — flood+fire only
        "coastal_moderate":   7.0,   # Rotterdam, Hamburg — storm surge added
        "high_seismic":       6.0,   # Istanbul, Lisbon — earthquake dominant
        "industrial_zone":    6.5,   # near SEVESO-III installations
    }

    BASE_RATES = {
        # Internal biosafety failure rates (log10, per-procedure)
        # Calibrated for BSL-3 moderate-safety at lambda=500 proc/month
        # (0,4) Normal->Disaster = Natech: handled via natech_rate param
        (0, 1): 3.0,   # Normal   -> Minor      ~1e-3
        (0, 2): 6.5,   # Normal   -> Serious: spontaneous equipment failure ~3e-7
        (1, 0): 2.0,   # Minor    -> Normal     ~1e-2  (recovery)
        (1, 2): 3.5,   # Minor    -> Serious    ~3e-4  (raised from 3.0)
        (1, 4): 6.0,   # Minor    -> Disaster   ~1e-6  (raised from 5.0)
        (2, 0): 1.5,   # Serious  -> Normal     ~3e-2  (rapid recovery)
        (2, 1): 2.0,   # Serious  -> Minor      ~1e-2  (partial recovery)
        (2, 3): 3.0,   # Serious  -> Critical   ~1e-3  (raised from 2.5)
        (2, 4): 5.0,   # Serious  -> Disaster   ~1e-5  (raised from 4.0)
        (3, 2): 1.0,   # Critical -> Serious    ~1e-1  (emergency containment)
        (3, 4): 3.5,   # Critical -> Disaster   ~3e-4  (raised from 2.0=1%)
    }
 

    # EDGE_WEIGHTS: intervention contributions to each edge's log10 barrier.
    # Format: (i,j) -> (class, [(source, weight), ...])
    # Final barrier = BASE_RATE[i,j] + sum(weight * effect[source])
    # Classes: H=training only  HE=training+maint  E=maint only  EO=maint+inspection
    #
    # Weight design rationale:
    #   0.3 x train  : partial training benefit (edge also has non-training drivers)
    #   0.5 x train  : full training benefit (edge is training-primary)
    #   0.8 x maint  : strong maintenance coupling (stepwise escalation path)
    #   0.5-0.6 x maint+insp : shared credit — both interventions needed for resilience
    EDGE_WEIGHTS = {
        (0, 1): ("H",  [("train", 0.3)]),              # Normal->Minor:    training reduces entry rate
        # (0,4) Normal->Disaster = Natech: excluded from intervention system
        (1, 0): ("H",  [("train", 0.5)]),              # Minor->Normal:    training accelerates recovery
        (1, 4): ("HE", [("train", 0.3), ("maint", 0.5)]),  # Minor->Disaster:  dual barrier (class HE)
        (1, 2): ("E",  [("maint", 0.8)]),              # Minor->Serious:   maintenance-dominated
        (2, 3): ("E",  [("maint", 0.8)]),              # Serious->Critical: maintenance-dominated
        (2, 0): ("E",  [("maint", 0.8)]),              # Serious->Normal:  maintenance aids recovery
        (2, 4): ("EO", [("maint", 0.6), ("insp", 0.6)]),   # Serious->Disaster: dual barrier (class EO)
        (3, 4): ("EO", [("maint", 0.5), ("insp", 0.6)]),   # Critical->Disaster: dual barrier (class EO)
    }

    version="5.4"

    def __init__(self, training=50.0, maint_freq=1.0, maint_adh=0.8,
                 inspection=70.0, natech_rate=7.5):
        # natech_rate: site-specific Natech log10 barrier for S0->S4 bypass
        # Default 7.5 = European inland BSL-3 (Brussels reference)
        self.natech_log = natech_rate
        self.training   = training
        self.maint_freq = maint_freq
        self.maint_adh  = maint_adh
        self.inspection = inspection
        self.train_effect = self._training_effect()
        self.maint_effect = self._maintenance_effect()
        self.insp_effect  = self._inspection_effect()
        self.P = self._build_matrix()
        self._N: Optional[np.ndarray] = None

    def _training_effect(self) -> float:
        """
        Training effect: Hill function with diminishing returns.

        DeltaL_train(h) = 2.0 * (1 - 1/(1 + (h/50)^2.5))

        Returns:
            Log-risk improvement from training
        """
        return 2.0 * (1.0 - 1.0 / (1.0 + (self.training / 50.0) ** 2.5))

    def _maintenance_effect(self) -> float:
        """
        Maintenance effect: frequency x adherence with gap penalty.

        DeltaL_maint(f,a) = L_freq(f) * m(a) * g(a)
        L_freq(f) = 1.2 * (1 - 1/(1 + f^1.5))
        m(a)      = min(1.0, 1.2*a)
        g(a)      = exp(-3*(1-a))

        Returns:
            Log-risk improvement from maintenance
        """
        freq_effect = 1.2 * (1.0 - 1.0 / (1.0 + (self.maint_freq / 1.0) ** 1.5))
        adh_mult    = min(1.0, 1.2 * self.maint_adh)
        gap_penalty = np.exp(-3.0 * (1.0 - self.maint_adh))
        return freq_effect * adh_mult * gap_penalty

    def _inspection_effect(self) -> float:
        """
        Inspection effect: threshold at 70.

        DeltaL_insp(s) = 1.0 if s >= 70, else 0.0

        Returns:
            Log-risk improvement from inspection
        """        
        return 1.0 if self.inspection >= 70 else 0.0

    def _build_matrix(self) -> np.ndarray:
        """
        Build the 5x5 transition matrix P.
        
        For each edge (i,j) with intervention contributions C:
            P[i,j] = 10^{ -(BASE_RATES[i,j] + sum(w * effect[src] for w, src in C)) }
        
        Diagonal P[i,i] = 1 - sum of off-diagonal row entries (ensures row-stochastic).
        P[4,4] = 1 (S4 absorbing).
        
        Edges not in EDGE_WEIGHTS receive no intervention increment (uncontrolled).
        """
        P   = np.zeros((5, 5))
        eff = {}
        def add(edge, amount):
            eff[edge] = eff.get(edge, 0.0) + amount
        effect_map = {"train": self.train_effect,
                      "maint": self.maint_effect,
                      "insp":  self.insp_effect}
        for edge, (_, contributions) in self.EDGE_WEIGHTS.items():
            for source, weight in contributions:
                add(edge, weight * effect_map[source])
        for (i, j), base in self.BASE_RATES.items():
            P[i, j] = 10.0 ** (-(base + eff.get((i, j), 0.0)))
        # Natech bypass: S0->S4 exogenous (no intervention effect)
        P[0, 4] = 10.0 ** (-self.natech_log)
        for i in range(4):
            P[i, i] = max(0.0, 1.0 - sum(P[i, k] for k in range(5) if k != i))
        P[4, 4] = 1.0
        return P

    @property
    def N(self) -> np.ndarray:
        """Fundamental matrix N = (I - Q)^{-1}."""
        if self._N is None:
            self._N = np.linalg.inv(np.eye(4) - self.P[:4, :4])
        return self._N

    def mfpt(self, start_state: int) -> float:
        """Full-chain MFPT to S4 from start_state (via fundamental matrix)."""
        return float(self.N.sum(axis=1)[start_state])

    def log_mfpt(self, start_state: int) -> float:
        return np.log10(self.mfpt(start_state))

    def conditional_mfpt(self, start_state: int) -> float:
        """
        Conditional MFPT to S4 from start_state, restricted to states
        >= start_state.  Probability mass that would de-escalate below
        start_state is removed and rows are renormalised, isolating the
        forward-escalation channel for that intervention class.

        This makes the three metrics genuinely distinct:
          L       = log10(cMFPT(S0)) — sensitive to training (S0->S1 barrier)
          logM_S1 = log10(cMFPT(S1)) — sensitive to maintenance (S1->S2 barrier)
          logM_S2 = log10(cMFPT(S2)) — sensitive to inspection (S2->S4 barriers)
        """
        k = start_state
        n_trans = 4 - k
        Q = np.zeros((n_trans, n_trans))
        for ii, si in enumerate(range(k, 4)):
            row_kept = sum(self.P[si, sj] for sj in range(k, 5))
            if row_kept <= 0:
                continue
            for jj, sj in enumerate(range(k, 4)):
                Q[ii, jj] = self.P[si, sj] / row_kept
        N_sub = np.linalg.inv(np.eye(n_trans) - Q)
        return float(N_sub.sum(axis=1)[0])

    def risk_per_procedure(self) -> dict:
        p0 = np.zeros(5); p0[0] = 1.0
        p1 = p0 @ self.P
        dp = float(p1[4])
        return {
            "disaster_prob": dp,
            "log_risk":      -np.log10(max(dp, 1e-15)),
            "state_probs":   dict(zip(self.STATE_NAMES, p1)),
        }

    def cumulative_risk(self, n: int = 10_000) -> float:
        """Probability of at least one disaster in n procedures."""
        return 1.0 - (1.0 - self.risk_per_procedure()["disaster_prob"]) ** n

    def metrics(self, lambda_proc: float = 1.0) -> dict:
        """
        Six primary risk metrics, optionally rescaled to calendar time.
    
        lambda_proc : procedure rate (procedures per calendar time unit).
                      Default 1.0 = nondimensional (procedure-scale).
                      Set to e.g. 500 for a facility running 500 proc/month
                      to obtain monthly MFPT and monthly log-risk.
    
        All three metrics use CONDITIONAL MFPT (restricted sub-chains):
          L       = log10(cMFPT(S0)) — S0 sub-chain, training channel
          logM_S1 = log10(cMFPT(S1)) — S1 sub-chain, maintenance channel
          logM_S2 = log10(cMFPT(S2)) — S2 sub-chain, inspection channel
        Each metric is sensitive to its own intervention and insensitive
        to others, enabling clean three-axis ROI decomposition.
    
        lambda_proc : procedure rate for calendar-time rescaling.
                      cMFPT_calendar = cMFPT_proc / lambda_proc.
        p_disaster  : raw one-step disaster probability from S0 (for simulation).
        state_probs : one-step distribution from S0 (["S1_Minor"] = near-miss rate).
        """
        r = self.risk_per_procedure()
        # Conditional MFPT: restricted sub-chains isolate each intervention channel
        cmfpt_s0 = self.conditional_mfpt(0) / lambda_proc
        cmfpt_s1 = self.conditional_mfpt(1) / lambda_proc
        cmfpt_s2 = self.conditional_mfpt(2) / lambda_proc
        return {
            "L":           np.log10(cmfpt_s0),   # training channel (S0 sub-chain)
            "p_disaster":  r["disaster_prob"],
            "MFPT_S1":     cmfpt_s1,
            "logM_S1":     np.log10(cmfpt_s1),   # maintenance channel (S1 sub-chain)
            "MFPT_S2":     cmfpt_s2,
            "logM_S2":     np.log10(cmfpt_s2),   # inspection channel (S2 sub-chain)
            "state_probs": r["state_probs"],
        }


    def print_status(self):
        r  = self.risk_per_procedure()
        mt = self.metrics()
        print(f"\n{chr(61)*60}\nBIOSAFETY LOG-RISK ASSESSMENT\n{chr(61)*60}")
        print(f"  Training:         {self.training:.0f} hours")
        print(f"  Maintenance freq: {self.maint_freq:.1f}/month")
        print(f"  Maintenance adh:  {self.maint_adh:.0%}")
        print(f"  Inspection:       {self.inspection:.0f}/100")
        print(f"  L   (log-risk S0):  {mt['L']:.3f}")
        print(f"  logM_S1 (MFPT S1):  {mt['logM_S1']:.3f}")
        print(f"  logM_S2 (MFPT S2):  {mt['logM_S2']:.3f}")
        print(f"  Disaster/procedure: {r['disaster_prob']:.3e}")
        print(f"  Cumulative (10k):   {self.cumulative_risk(10000):.3e}")

    def print_edge_classes(self):
        print("\nEdge    Class   Base-rate   Intervention weights")
        print("-" * 58)
        for edge, cls, base, w in [
            ("(0,1)", "H",   "3.0", "0.3 x train"),
            ("(1,0)", "H",   "2.0", "0.5 x train"),
            ("(1,4)", "HE",  "6.0", "0.3 x train  +  0.5 x maint"),
            ("(1,2)", "E",   "3.5", "0.8 x maint"),
            ("(2,0)", "E",   "1.5", "0.8 x maint"),
            ("(2,3)", "E",   "3.0", "0.8 x maint"),
            ("(2,4)", "EO",  "5.0", "0.6 x maint  +  0.6 x insp"),
            ("(3,4)", "EO",  "3.5", "0.5 x maint  +  0.6 x insp"),
            ("(2,1)", "-",   "2.0", "[uncontrolled]"),
            ("(3,2)", "-",   "1.0", "[uncontrolled]"),
        ]:
            print(f"  {edge}    {cls:<4}    {base:<8}   {w}")
        print(f"  (0,4)    NATECH  {self.natech_log:<8}   [site-specific exogenous]")

    @classmethod
    def from_risk_group(cls, rg, training=20.0, maint_freq=0.5,
                        maint_adh=0.5, inspection=50.0):
        rg_priors = {1: 7.0, 2: 5.5, 3: 4.0, 4: 2.0}
        m = cls(training=training, maint_freq=maint_freq,
                maint_adh=maint_adh, inspection=inspection)
        print(f"RG{rg} prior anchor: L ~ {rg_priors[rg]:.1f}  |  Model L = {m.metrics()['L']:.2f}")
        return m


# ── Simulation ───────────────────────────────────────────────────────
def simulate_observations(model, n_months=36, procedures_per_month=500, seed=None):
    """
    Simulate monthly counts of three observable event types:
      near_misses    : S0->S1 transitions       (rate ~ P[0,1])
      light_incidents: S2->S3 escalations        (rate ~ P[2,3])
      disasters      : absorption into S4        (rate ~ disaster_prob)

    Default 500 procedures/month targets ~1 near-miss and
    ~1 light incident per 1,000 procedures for a moderate-safety facility.
    """
    if seed is not None:
        np.random.seed(seed)
    r   = model.risk_per_procedure()
    p01 = r["state_probs"]["S1_Minor"]
    p23 = float(model.P[2, 3])
    data = []
    for month in range(1, n_months + 1):
        data.append({
            "month":           month,
            "near_misses":     int(np.random.binomial(procedures_per_month, p01)),
            "light_incidents": int(np.random.binomial(procedures_per_month, p23)),
            "disasters":       int(np.random.binomial(procedures_per_month, r["disaster_prob"])),
        })
    df = pd.DataFrame(data)
    df.attrs["ppm"] = procedures_per_month
    return df


# ── MCMC ─────────────────────────────────────────────────────────────
class MCMCBiosafety:
    """
    Metropolis-Hastings MCMC with three-observable likelihood:
      (1) near-misses    ~ Binomial(n, P[0,1])
      (2) light incidents ~ Binomial(n, P[2,3])
      (3) disasters       ~ Binomial(n, disaster_prob)

    The light-incident channel (S2->S3 escalations) substantially
    improves identifiability of maintenance and inspection parameters
    compared to a disaster-only likelihood.
    """

    def __init__(self, initial_model, observed_data):
        self.initial = initial_model
        ppm          = observed_data.attrs.get("ppm", 500)
        self.n_proc  = len(observed_data) * ppm
        self.obs_nm  = int(observed_data["near_misses"].sum())
        self.obs_li  = int(observed_data["light_incidents"].sum())
        self.obs_d   = int(observed_data["disasters"].sum())

    def log_prior(self, h, f, a, s) -> float:
        if h <= 0 or f <= 0 or not (0 <= a <= 1) or not (0 <= s <= 100):
            return -np.inf
        return (stats.lognorm.logpdf(h, s=0.5, scale=50) +
                stats.lognorm.logpdf(f, s=0.5, scale=1.0) +
                stats.beta.logpdf(a, 2, 2) -
                np.log(101))

    def log_likelihood(self, h, f, a, s) -> float:
        m   = BiosafetyModel(training=h, maint_freq=f, maint_adh=a, inspection=s)
        r   = m.risk_per_procedure()
        p23 = float(m.P[2, 3])
        return (
            stats.binom.logpmf(self.obs_nm, self.n_proc, r["state_probs"]["S1_Minor"]) +
            stats.binom.logpmf(self.obs_li, self.n_proc, p23) +
            stats.binom.logpmf(self.obs_d,  self.n_proc, r["disaster_prob"])
        )

    def run(self, n_iter=20_000, burnin=5_000, thin=5, seed=None) -> pd.DataFrame:
        if seed is not None:
            np.random.seed(seed)
        cur = {
            "h": self.initial.training,
            "f": self.initial.maint_freq,
            "a": self.initial.maint_adh,
            "s": float(self.initial.inspection),
        }
        scale    = {"h": 5.0, "f": 0.2, "a": 0.1, "s": 3.0}
        cur_post = self.log_likelihood(**cur) + self.log_prior(**cur)
        samples, accepted = [], 0
        for i in range(n_iter):
            prop = {
                "h": max(0.1, cur["h"] + np.random.normal(0, scale["h"])),
                "f": max(0.1, cur["f"] + np.random.normal(0, scale["f"])),
                "a": float(np.clip(cur["a"] + np.random.normal(0, scale["a"]), 0, 1)),
                "s": float(np.clip(round(cur["s"] + np.random.normal(0, scale["s"])), 0, 100)),
            }
            pp = self.log_likelihood(**prop) + self.log_prior(**prop)
            if np.log(np.random.random()) < pp - cur_post:
                cur, cur_post = prop, pp
                accepted += 1
            if i >= burnin and i % thin == 0:
                m  = BiosafetyModel(training=cur["h"], maint_freq=cur["f"],
                                    maint_adh=cur["a"], inspection=cur["s"])
                mt = m.metrics()
                samples.append({
                    "training":    cur["h"],
                    "maint_freq":  cur["f"],
                    "maint_adh":   cur["a"],
                    "inspection":  cur["s"],
                    "L":           mt["L"],
                    "logM_S1":     mt["logM_S1"],
                    "logM_S2":     mt["logM_S2"],
                    "disaster_prob": mt["p_disaster"],
                })
        self.acceptance_rate = accepted / n_iter
        print(f"Acceptance rate: {self.acceptance_rate:.3f}")
        return pd.DataFrame(samples)


# ── Cost functions ────────────────────────────────────────────────────
def intervention_cost(interv, cur, new) -> float:
    """
    Marginal cost of changing intervention level from cur to new.
      training   : $100/hour
      maint_freq : $10,000/unit/month
      maint_adh  : $20,000 x delta x (1 + 2*cur)  [convex]
      inspection : $30,000 one-time if crossing threshold 70
    """
    if interv == "training":   return abs(new - cur) * 100
    if interv == "maint_freq": return abs(new - cur) * 10_000
    if interv == "maint_adh":  return 20_000 * (new - cur) * (1 + 2 * cur) if new > cur else 0.0
    if interv == "inspection": return 30_000.0 if cur < 70 <= new else 0.0
    return 0.0


# ── ROI functions ─────────────────────────────────────────────────────
def training_roi(base, new_training, N_annual=10_000, V=10_000_000) -> dict:
    """Monetary ROI for training via per-procedure log-risk L."""
    bm   = base.metrics()
    new  = BiosafetyModel(training=new_training, maint_freq=base.maint_freq,
                          maint_adh=base.maint_adh, inspection=base.inspection)
    nm   = new.metrics()
    cost = intervention_cost("training", base.training, new_training)
    ben  = (N_annual / 10**bm["L"] - N_annual / 10**nm["L"]) * V
    return {
        "cost":       cost,
        "benefit_yr": ben,
        "roi":        ben / cost if cost > 0 else float("inf"),
        "payback_yr": cost / ben if ben > 0 else float("inf"),
        "L_new":      nm["L"],
    }


def maintenance_roi_log(base, new_freq, new_adh) -> dict:
    """Dimensionless ROI: DeltaLogM_S1 per $1,000 invested."""
    bm   = base.metrics()
    new  = BiosafetyModel(training=base.training, maint_freq=new_freq,
                          maint_adh=new_adh, inspection=base.inspection)
    nm   = new.metrics()
    cost  = (intervention_cost("maint_freq", base.maint_freq, new_freq) +
             intervention_cost("maint_adh",  base.maint_adh,  new_adh))
    delta = nm["logM_S1"] - bm["logM_S1"]
    return {
        "cost":          cost,
        "delta_logM_S1": delta,
        "log_per_1k":    delta / (cost / 1000) if cost > 0 else float("inf"),
        "logM_S1_new":   nm["logM_S1"],
    }


def inspection_roi_log(base, new_inspection) -> dict:
    """Dimensionless ROI: DeltaLogM_S2 per $1,000 invested."""
    bm   = base.metrics()
    new  = BiosafetyModel(training=base.training, maint_freq=base.maint_freq,
                          maint_adh=base.maint_adh, inspection=new_inspection)
    nm   = new.metrics()
    cost  = intervention_cost("inspection", base.inspection, new_inspection)
    delta = nm["logM_S2"] - bm["logM_S2"]
    return {
        "cost":          cost,
        "delta_logM_S2": delta,
        "log_per_1k":    delta / (cost / 1000) if cost > 0 else float("inf"),
        "logM_S2_new":   nm["logM_S2"],
    }


def optimal_allocation(base, budget=100_000, N_annual=10_000,
                        V=10_000_000, scale_serious=10.0) -> dict:
    """Grid search for optimal allocation within budget."""
    bm        = base.metrics()
    ann_minor = N_annual * base.P[0, 1]
    ann_ser   = ann_minor * base.P[1, 2] * scale_serious
    best_b, best = -np.inf, {}
    for th in [0, 20, 40, 60, 80, 100, 120]:
        for mf in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            for ma in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                for ins in [50, 60, 70, 75, 80, 85, 90]:
                    tc = (intervention_cost("training",   base.training,   th) +
                          intervention_cost("maint_freq", base.maint_freq, mf) +
                          intervention_cost("maint_adh",  base.maint_adh,  ma) +
                          intervention_cost("inspection", base.inspection, ins))
                    if tc > budget:
                        continue
                    m  = BiosafetyModel(training=th, maint_freq=mf, maint_adh=ma, inspection=ins)
                    mt = m.metrics()
                    b_t = (N_annual / 10**bm["L"] - N_annual / 10**mt["L"]) * V
                    b_m = ann_minor * (1/bm["MFPT_S1"] - 1/mt["MFPT_S1"]) * (-V)
                    b_i = ann_ser   * (1/bm["MFPT_S2"] - 1/mt["MFPT_S2"]) * (-V)
                    b   = b_t + b_m + b_i
                    if b > best_b:
                        best_b = b
                        best = {
                            "training": th, "maint_freq": mf,
                            "maint_adh": ma, "inspection": ins,
                            "cost": tc, "benefit_train": b_t,
                            "benefit_maint": b_m, "benefit_insp": b_i,
                            "total_benefit": b, "roi": b / budget,
                            "L": mt["L"], "logM_S1": mt["logM_S1"], "logM_S2": mt["logM_S2"],
                        }
    return best


# ══════════════════════════════════════════════════════════════════════
def demonstrate():
    """
    Full demonstration: priors, maintenance scenarios, MCMC (Scenario 2),
    three-metric ROI framework, edge class table.

    MCMC uses a moderate-safety facility (BSL-3 analogue, training=35h,
    freq=0.8, adh=60%, inspection=55) with three observables:
    near-misses, light incidents (S2->S3), and disasters.
    500 procedures/month over 36 months = 18,000 procedures total.
    """
    N_annual = 10_000; 
    V = 50_000_000; SCALE = 10
    SEP = "=" * 70

    print(f"\n{SEP}\nBIOSAFETY LOG-RISK MARKOV FRAMEWORK  v{version}\n{SEP}")
    print("Three-metric ROI: L (training) | logM_S1 (maintenance) | logM_S2 (inspection)")

    # [1] Priors
    np.random.seed(42)
    print("\n[1] PRIOR DISTRIBUTIONS")
    h_s = np.random.lognormal(np.log(50), 0.5, 10000)
    f_s = np.random.lognormal(np.log(1.0), 0.5, 10000)
    a_s = np.random.beta(2, 2, 10000)
    print(f"  Training hours:   median={np.median(h_s):.1f}  "
          f"95% CI=[{np.percentile(h_s,2.5):.1f}, {np.percentile(h_s,97.5):.1f}]")
    print(f"  Maintenance freq: median={np.median(f_s):.2f}  "
          f"95% CI=[{np.percentile(f_s,2.5):.2f}, {np.percentile(f_s,97.5):.2f}]")
    print(f"  Maintenance adh:  mean={np.mean(a_s):.2f}   "
          f"95% CI=[{np.percentile(a_s,2.5):.2f}, {np.percentile(a_s,97.5):.2f}]")

    # [2] Maintenance scenarios
    print("\n[2] MAINTENANCE: FREQUENCY vs ADHERENCE  (logM_S1, training=50h, inspection=75)")
    scenarios = [
        ("Low freq, poor adh",  0.5, 0.4),
        ("High freq, poor adh", 2.0, 0.4),
        ("Low freq, good adh",  0.5, 0.9),
        ("Moderate, perfect",   1.0, 1.0),
        ("High freq, good adh", 2.0, 0.9),
    ]
    print(f"  {chr(45)*66}")
    print(f"  {'Scenario':<22} {'Freq':>5} {'Adh':>5} {'Maint':>6} {'MFPT_S1':>12} {'logM_S1':>9} {'L':>6}")
    print(f"  {chr(45)*66}")
    ms1 = []
    for name, f, a in scenarios:
        m  = BiosafetyModel(training=50, maint_freq=f, maint_adh=a, inspection=75)
        mt = m.metrics()
        ms1.append(mt["MFPT_S1"])
        print(f"  {name:<22} {f:>5.1f} {a:>5.1f} {m.maint_effect:>6.2f} "
              f"{mt['MFPT_S1']:>12,.0f} {mt['logM_S1']:>9.2f} {mt['L']:>6.2f}")
    print(f"  logM_S1 spread={np.log10(ms1[4]/ms1[0]):.2f}  "
          f"improvement={ms1[4]/ms1[0]:.1f}x")
    print(f"  Adherence > Frequency: low/good ({np.log10(ms1[2]):.2f}) "
          f"> high/poor ({np.log10(ms1[1]):.2f})")

    # [3] MCMC — Scenario 2 (moderate-safety, three observables)
    LAMBDA = 500   # procedures/month — facility throughput assumption
    print("\n[3] BAYESIAN UPDATING — Scenario 2 (moderate-safety, BSL-3 analogue)")
    print(f"    Observables: near-misses (S0->S1), light incidents (S2->S3), disasters")
    print(f"    lambda={LAMBDA} proc/month  x  36 months  =  {LAMBDA*36:,} procedures")
    print(f"    Time unit for MFPT: months (MFPT_procedures / {LAMBDA})")
    true_m   = BiosafetyModel(training=35, maint_freq=0.8, maint_adh=0.6, inspection=55)
    true_mt  = true_m.metrics(lambda_proc=LAMBDA)          # <-- calendar-scale
    observed = simulate_observations(true_m, n_months=36,
                                     procedures_per_month=LAMBDA, seed=42)
    init_m   = BiosafetyModel(training=50, maint_freq=1.0, maint_adh=0.7, inspection=60)
    print(f"  True: training={true_m.training:.0f}h  freq={true_m.maint_freq:.2f}  "
          f"adh={true_m.maint_adh:.0%}  insp={true_m.inspection:.0f}")
    print(f"  L={true_mt['L']:.2f} (monthly)  "
          f"logM_S1={true_mt['logM_S1']:.2f} (months)  "
          f"logM_S2={true_mt['logM_S2']:.2f} (months)")
    print(f"  Data: near_misses={observed['near_misses'].sum()}  "
          f"light_incidents={observed['light_incidents'].sum()}  "
          f"disasters={observed['disasters'].sum()}")

    
    
    mcmc = MCMCBiosafety(init_m, observed)
    post = mcmc.run(n_iter=20000, burnin=5000, thin=5, seed=99)
    print(f"  {'Parameter':<12} {'True':>8} {'Mean':>8} {'Std':>7} "
          f"{'2.5%':>8} {'97.5%':>8}  {'CI covers?':>10}")
    print(f"  {'(monthly scale where applicable)'}")
    print(f"  {chr(45)*62}")
    for name, tv, col in [
        ("training",   true_m.training,    post["training"]),
        ("maint_freq", true_m.maint_freq,  post["maint_freq"]),
        ("maint_adh",  true_m.maint_adh,   post["maint_adh"]),
        ("inspection", true_m.inspection,  post["inspection"]),
        ("L (monthly)",      true_mt["L"],       post["L"] - np.log10(LAMBDA)),
        ("logM_S1 (mo)",     true_mt["logM_S1"], post["logM_S1"] - np.log10(LAMBDA)),
        ("logM_S2 (mo)",     true_mt["logM_S2"], post["logM_S2"] - np.log10(LAMBDA)),
    ]:
        lo, hi = col.quantile(0.025), col.quantile(0.975)
        covers = "YES" if lo <= tv <= hi else "NO"
        print(f"  {name:<12} {tv:>8.2f} {col.mean():>8.2f} {col.std():>7.2f} "
              f"{lo:>8.2f} {hi:>8.2f}  {covers:>10}")

    # [4] ROI
    base  = BiosafetyModel(training=20, maint_freq=0.5, maint_adh=0.5, inspection=50)
    bm    = base.metrics()
    ann_m = N_annual * base.P[0, 1]
    ann_s = ann_m * base.P[1, 2] * SCALE
    print(f"\n[4] ROI — THREE-METRIC FRAMEWORK")
    print(f"  Baseline: training=20h  freq=0.5  adh=50%  insp=50")
    print(f"  L={bm['L']:.2f}  logM_S1={bm['logM_S1']:.2f}  logM_S2={bm['logM_S2']:.2f}")
    print(f"  Minor/yr={ann_m:.1f}  Serious/yr={ann_s:.3f}  "
          f"DisasterCost=${N_annual/10**bm['L']*V:,.0f}/yr")

    print("\n  [4a] Training ROI (monetary, metric: L)")
    for h in [40, 60, 80, 100, 120]:
        r  = training_roi(base, h, N_annual, V)
        pb = f"{r['payback_yr']:.2f}yr" if r['payback_yr'] < 100 else "inf"
        print(f"  {h:>3}h  cost=${r['cost']:>6,}  ben=${r['benefit_yr']:>9,.0f}/yr  "
              f"ROI={r['roi']:>6.1f}x  payback={pb:>8}  L={r['L_new']:.2f}")

    print("\n  [4b] Maintenance ROI (dimensionless: DeltaLogM_S1 per $1k)")
    for mf, ma in [(0.5, 0.9), (1.0, 0.8), (1.5, 0.8), (2.0, 0.9), (2.0, 0.95)]:
        r = maintenance_roi_log(base, mf, ma)
        print(f"  freq={mf:.1f} adh={ma:.2f}  cost=${r['cost']:>7,}  "
              f"DlogM_S1={r['delta_logM_S1']:>+7.3f}  "
              f"Dlog/$1k={r['log_per_1k']:.4f}")

    print("\n  [4c] Inspection ROI (dimensionless: DeltaLogM_S2 per $1k)")
    print("  NOTE: threshold-saturated above score 70 — one-time $30k compliance investment")
    for ins in [70, 75, 80, 90]:
        r = inspection_roi_log(base, ins)
        print(f"  insp={ins}  cost=${r['cost']:>7,}  "
              f"DlogM_S2={r['delta_logM_S2']:>+7.3f}  "
              f"Dlog/$1k={r['log_per_1k']:.4f}")

    print("\n  [4d] Training sensitivity to V (80h from baseline)")
    for Vv in [1e6, 5e6, 10e6, 50e6, 100e6]:
        r  = training_roi(base, 80, N_annual, Vv)
        pb = f"{r['payback_yr']:.2f}yr" if r['payback_yr'] < 100 else "inf"
        print(f"  V=${Vv/1e6:.0f}M  ben=${r['benefit_yr']:>10,.0f}/yr  "
              f"ROI={r['roi']:>7.1f}x  payback={pb:>8}")

    print("\n  [4e] Optimal $100k allocation")
    best = optimal_allocation(base, budget=100_000, N_annual=N_annual,
                               V=V, scale_serious=SCALE)
    print(f"  training={best['training']:.0f}h  freq={best['maint_freq']:.2f}  "
          f"adh={best['maint_adh']:.0%}  insp={best['inspection']:.0f}")
    roi_actual = best['total_benefit'] / best['cost'] if best['cost'] > 0 else 0
    print(f"  cost=${best['cost']:,}  total=${best['total_benefit']:,.0f}/yr  ROI={roi_actual:.1f}x")
    
    ann_b = N_annual / 10**bm["L"] * V
    ann_a = N_annual / 10**best["L"] * V
    print(f"  Disaster cost: ${ann_b:,.0f} -> ${ann_a:,.0f}/yr  "
          f"({(1 - ann_a/ann_b)*100:.1f}% reduction)")

    # [5] Edge table
    print("\n[5] EDGE CLASS TABLE")
    BiosafetyModel().print_edge_classes()


if __name__ == "__main__":
    demonstrate()