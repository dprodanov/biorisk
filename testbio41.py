# -*- coding: utf-8 -*-
"""

@author: Dimiter Prodanov
IICT-BAS, BioPrecis

Biosafety Log-Risk Markov Framework
====================================
Implementation of the methodology described in:
"Making Biorisk Measurable: A Bayesian Framework for Laboratory Risk Management"

Version: 4.0 - with cause-class tagging
date 23 Feb 2026


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ============================================================================
# BIOHAZARD MODIFICATION
# ============================================================================

class HazardModifier:
    RG_PRIORS = {1: 7.0, 2: 5.5, 3: 4.0, 4: 2.0}  # Log-risk means (higher=safer)

    # Quantitative boundaries between RG classes (log10 risk)
    T12 = 6.25  # RG1 / RG2
    T23 = 4.75  # RG2 / RG3  
    T34 = 3.00  # RG3 / RG4

    @classmethod
    def classify_from_logrisk(cls, L: float) -> int:
        """Map log-risk L to WHO-like RG (1–4)."""
        if L >= cls.T12:
            return 1
        elif L >= cls.T23:
            return 2
        elif L >= cls.T34:
            return 3
        else:
            return 4

    @classmethod
    def from_genetic_mods(cls, parent_rg: int,
                          mf: float = 1.0,
                          tp: float = 1.0) -> tuple[int, float]:
        if parent_rg not in cls.RG_PRIORS:
            raise ValueError("parent_rg must be 1-4")

        l_parent = cls.RG_PRIORS[parent_rg]
        k = mf * tp  # risk multiplier >1
        l_adjusted = l_parent - np.log10(k)
        l_prior = max(0.0, l_adjusted)

        adjusted_rg = cls.classify_from_logrisk(l_prior)

        print(f"Parent RG{parent_rg} L={l_parent:.2f} → "
              f"Adjusted L={l_prior:.2f} (RG{adjusted_rg})")
        return adjusted_rg, l_prior



# ============================================================================
# CORE BIORISK MODEL: 5-STATE MARKOV CHAIN WITH INTERVENTIONS
# ============================================================================

class BiosafetyModel:
    """
    5-state Markov chain for laboratory biosafety.

    States:
        S0: Normal operation (all systems functional)
        S1: Minor issues (procedural deviations, fatigue)
        S2: Serious issues (equipment malfunction)
        S3: Critical (containment threatened)
        S4: DISASTER (actual release/exposure) - absorbing

    Interventions:
        - Training:    diminishing returns (Hill function)    [H-class edges]
        - Maintenance: frequency x adherence + gap penalty    [E-class edges]
        - Inspection:  threshold effect at 70/100             [O-class edges]

    Cause classes:
        H: Human performance (procedural deviations, fatigue)
        E: Equipment/engineering (malfunction, technical failure)
        O: Organizational (supervision, inspection, safety culture)
    """

    STATE_NAMES = ['S0_Normal', 'S1_Minor', 'S2_Serious', 'S3_Critical', 'S4_Disaster']

    def __init__(self,
                 training:   float = 50,   # training hours
                 maint_freq: float = 1.0,  # scheduled frequency (per month)
                 maint_adh:  float = 0.8,  # adherence rate (0-1)
                 inspection: int   = 70):  # inspection score (0-100)
    
    
        if training < 0:
            raise ValueError("Training hours cannot be negative.")
        if maint_freq < 0:
            raise ValueError("Maintenance frequency cannot be negative.")
        if inspection < 0:
            raise ValueError("Inspection hours cannot be negative.")

        # Base transition rates (log10 scale) - calibrated to WHO RG-3 baseline
        self.base_rates = {
            (0,1): 3.0, (0,4): 6.0,                           # Normal
            (1,0): 2.0, (1,2): 3.0, (1,4): 5.0,               # Minor
            (2,0): 1.5, (2,1): 2.0, (2,3): 2.5, (2,4): 4.0,  # Serious
            (3,2): 1.0, (3,4): 2.0,                           # Critical
        }

        # Cause-class tagging of transitions (see manuscript)
        # H: training acts here
        # E: maintenance acts here
        # O: inspection acts here
        self.edge_class = {
            (0,1): "H",   # procedural deviation onset
            (0,4): "H",   # catastrophic human-path failure
            (1,4): "H",   # deviation leads to exposure
            (1,0): "O",   # recovery: organizational response restores normal
            (1,2): "E",   # equipment malfunction becomes salient
            (2,3): "E",   # escalation of technical failure
            (2,4): "E",   # technical failure leads to exposure
            (3,4): "E",   # containment integrity failure -> disaster
            (2,0): "O",   # corrective action returns to normal
            (2,1): "O",   # partial de-escalation (organizational)
            (3,2): "O",   # stabilization via programmatic response
        }

        self.training   = training
        self.maint_freq = maint_freq
        self.maint_adh  = maint_adh
        self.inspection = inspection

        self.train_effect = self._training_effect()
        self.maint_effect = self._maintenance_effect()
        self.insp_effect  = self._inspection_effect()

        self.P = self._build_matrix()

    # ------------------------------------------------------------------
    # Cause-class enforcement
    # ------------------------------------------------------------------

    def allowed(self, edge: tuple, mech: str) -> bool:
        """
        Cause-class sparsity rule:
            train -> H-class edges only
            maint -> E-class edges only
            insp  -> O-class edges only
        """
        c = self.edge_class.get(edge, None)
        if mech == "train": return c == "H"
        if mech == "maint": return c == "E"
        if mech == "insp":  return c == "O"
        return False

    # ------------------------------------------------------------------
    # Intervention effect functions
    # ------------------------------------------------------------------

    def _training_effect(self) -> float:
        """
        Training effect: Hill function with diminishing returns.

        DeltaL_train(h) = 2.0 * (1 - 1/(1 + (h/50)^2.5))

        Returns:
            Log-risk improvement from training
        """
        max_effect = 2.0
        half_sat   = 50.0
        shape      = 2.5
        return max_effect * (1 - 1/(1 + (self.training/half_sat)**shape))

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
        freq_effect = 1.2 * (1 - 1/(1 + (self.maint_freq/1.0)**1.5))
        adh_mult    = min(1.0, 1.2 * self.maint_adh)
        gap_penalty = np.exp(-3.0 * (1 - self.maint_adh))
        return freq_effect * adh_mult * gap_penalty

    def _inspection_effect(self) -> float:
        """
        Inspection effect: threshold at 70.

        DeltaL_insp(s) = 1.0 if s >= 70, else 0.0

        Returns:
            Log-risk improvement from inspection
        """
        return 1.0 if self.inspection >= 70 else 0.0

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def _build_matrix(self) -> np.ndarray:
        """
        Build 5x5 row-stochastic transition probability matrix.

        Intervention effects are allocated strictly by cause class:
            H-edges: training only
            E-edges: maintenance only
            O-edges: inspection only

        Raises ValueError if an allocation violates the class rule.

        Returns:
            5x5 row-stochastic matrix
        """
        P       = np.zeros((5, 5))
        effects = {}

        def add(edge, mech, amount):
            if not self.allowed(edge, mech):
                raise ValueError(
                    f"Intervention '{mech}' not allowed on edge {edge} "
                    f"(cause class '{self.edge_class.get(edge, 'undefined')}')"
                )
            effects[edge] = effects.get(edge, 0.0) + amount

        # H-class: training only
        add((0,1), "train", 0.3 * self.train_effect)
        add((0,4), "train", 0.5 * self.train_effect)
        add((1,4), "train", 0.3 * self.train_effect)

        # E-class: maintenance only
        add((1,2), "maint", 0.8 * self.maint_effect)
        add((2,3), "maint", 0.8 * self.maint_effect)
        add((2,4), "maint", 0.6 * self.maint_effect)
        add((3,4), "maint", 0.5 * self.maint_effect)

        # O-class: inspection only
        add((1,0), "insp", 1.0 * self.insp_effect)
        add((2,0), "insp", 1.0 * self.insp_effect)
        add((3,2), "insp", 1.0 * self.insp_effect)

        # Fill matrix from base rates + effects
        for (i,j), base in self.base_rates.items():
            effect = effects.get((i,j), 0.0)
            P[i,j] = 10 ** (-(base + effect))

        # Diagonal: self-retention
        for i in range(4):
            row_sum = sum(P[i,j] for j in range(5) if j != i)
            P[i,i]  = max(0.0, 1 - row_sum)

        # Absorbing disaster state
        P[4,4] = 1.0
        return P

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def risk_per_procedure(self) -> Dict:
        """
        Calculate disaster probability for one procedure.

        Starting in S0 (normal operation), compute state distribution
        after one step.

        Returns:
            Dictionary with disaster_prob, log_risk, and state_probs
        """
        p0    = np.zeros(5)
        p0[0] = 1.0
        p1    = p0 @ self.P

        disaster_prob = p1[4]
        log_risk      = -np.log10(max(disaster_prob, 1e-15))

        return {
            'disaster_prob': disaster_prob,
            'log_risk':      log_risk,
            'state_probs':   dict(zip(self.STATE_NAMES, p1))
        }

    def cumulative_risk(self, n_procedures: int = 10000) -> float:
        """
        Calculate cumulative disaster risk over multiple procedures.

        Args:
            n_procedures: Number of procedures (default 10,000 = annual)

        Returns:
            Probability of at least one disaster
        """
        p_single = self.risk_per_procedure()['disaster_prob']
        return 1 - (1 - p_single) ** n_procedures

    def scenario_risk(self, n_steps: int = 10) -> Dict:
        """
        Disaster probability over n_steps from S0.
    
        Used for illustrative multi-step ROI only — not in
        Bayesian updating or per-procedure risk reporting.
    
        Args:
            n_steps: Number of Markov steps (default 10)
    
        Returns:
            Dictionary with disaster_prob, log_risk, state_probs
        """
        p = np.zeros(5)
        p[0] = 1.0
        for _ in range(n_steps):
            p = p @ self.P
    
        disaster_prob = p[4]
        log_risk      = -np.log10(max(disaster_prob, 1e-15))
    
        return {
            'disaster_prob': disaster_prob,
            'log_risk':      log_risk,
            'state_probs':   dict(zip(self.STATE_NAMES, p))
        }


    @classmethod
    def from_risk_group(cls,
                        rg: int,
                        training: float = 40,
                        maint_freq: float = 0.8,
                        maint_adh: float = 0.6,
                        inspection: int = 55) -> 'BiosafetyModel':
        """
        Initialise model with WHO Risk Group prior, using fixed log-risk anchors.

        rg_priors[rg] gives the *conceptual* prior mean log-risk L_RG.
        The actual model log-risk L_model is reported so users can see
        how close the chosen intervention profile is to the anchor.

        Args:
            rg: WHO Risk Group (1-4)
            training, maint_freq, maint_adh, inspection:
                Default intervention levels for the baseline model.

        Returns:
            BiosafetyModel whose log-risk is reported alongside the RG prior.
        """
        rg_priors = {1: 7.0, 2: 5.5, 3: 4.0, 4: 2.0}
        if rg not in rg_priors:
            raise ValueError("rg must be 1-4")

        target_L = rg_priors[rg]

        # Build model with the specified intervention profile
        model = cls(training=training,
                    maint_freq=maint_freq,
                    maint_adh=maint_adh,
                    inspection=inspection)

        risk = model.risk_per_procedure()
        L_model = risk["log_risk"]

        print(f"RG{rg} prior anchor: L ~ N({target_L:.1f}, 0.3)")
        print(f"Model with training={training:.0f}, "
              f"freq={maint_freq:.2f}/month, adh={maint_adh:.0%}, "
              f"insp={inspection}: L = {L_model:.2f}")

        return model

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_status(self):
        """Print formatted risk report."""
        risk   = self.risk_per_procedure()
        annual = self.cumulative_risk(10000)

        print("\n" + "="*60)
        print("BIOSAFETY LOG-RISK ASSESSMENT")
        print("="*60)
        print(f"\nInterventions:")
        print(f"  Training:           {self.training:.0f} hours")
        print(f"  Maintenance freq:   {self.maint_freq:.1f}/month")
        print(f"  Maintenance adh:    {self.maint_adh:.0%}")
        print(f"  Inspection:         {self.inspection}/100")
        print(f"\nEffects (log improvement):")
        print(f"  Training:           +{self.train_effect:.2f} logs")
        print(f"  Maintenance:        +{self.maint_effect:.2f} logs")
        print(f"  Inspection:         +{self.insp_effect:.2f} logs")
        print(f"\nRisk per procedure:")
        print(f"  Disaster probability: {risk['disaster_prob']:.2e}")
        print(f"  Log-risk score:       {risk['log_risk']:.2f}")
        print(f"\nAnnual risk (10,000 procedures):")
        print(f"  Disaster probability: {annual:.2e}")
        print(f"  Expected disasters:   {annual:.4f}")
        print("="*60)

    def print_edge_classes(self):
        """Print edge->cause-class table for manuscript appendix."""
        print("\nEdge    Class   Base-rate (log10)")
        print("-" * 35)
        for (i,j), c in sorted(self.edge_class.items()):
            base = self.base_rates.get((i,j), "—")
            print(f"  ({i},{j})    {c}       {base}")


# ============================================================================
# MCMC UPDATING
# ============================================================================

class MCMCBiosafety:
    """
    Bayesian updating of model parameters using observed data.

    Priors (manuscript Section 4.2):
        - Training:    LogNormal(median=50, scale=0.5)
        - Maint freq:  LogNormal(median=1.0, scale=0.5)
        - Maint adh:   Beta(2,2)
        - Inspection:  DiscreteUniform(0,100)
    """

    def __init__(self, initial_model: BiosafetyModel, observed_data: pd.DataFrame):
        """
        Args:
            initial_model: Starting point for MCMC
            observed_data: DataFrame with 'incidents' and 'near_misses' columns
        """
        self.initial        = initial_model
        self.data           = observed_data
        self.n_procedures   = len(observed_data) * 100  # 100 per month
        self.obs_disasters  = observed_data['incidents'].sum()
        self.obs_nearmisses = observed_data['near_misses'].sum()

    def log_prior(self, h: float, f: float, a: float, s: int) -> float:
        """
        Log-prior probability density.

        Args:
            h: training hours (>0)
            f: maintenance frequency (>0)
            a: maintenance adherence [0,1]
            s: inspection score [0,100] integer

        Returns:
            Log-prior density
        """
        logp = 0

        if h > 0:
            logp += stats.lognorm.logpdf(h, s=0.5, scale=50)
        else:
            return -np.inf

        if f > 0:
            logp += stats.lognorm.logpdf(f, s=0.5, scale=1.0)
        else:
            return -np.inf

        if 0 <= a <= 1:
            logp += stats.beta.logpdf(a, 2, 2)
        else:
            return -np.inf

        if 0 <= s <= 100 and isinstance(s, (int, np.integer)):
            logp -= np.log10(101)
        else:
            return -np.inf

        return logp

    def log_likelihood(self, h: float, f: float, a: float, s: int) -> float:
        """
        Log-likelihood of observed data given parameters.

        Disasters:   Binomial(N, p_disaster)
        Near-misses: Binomial(N, p_S2)  where p_S2 = one-step prob of S2
        """
        model = BiosafetyModel(training=h, maint_freq=f, maint_adh=a, inspection=s)
        risk  = model.risk_per_procedure()

        ll  = stats.binom.logpmf(
            self.obs_disasters,
            self.n_procedures,
            risk['disaster_prob']
        )
        ll += stats.binom.logpmf(
            self.obs_nearmisses,
            self.n_procedures,
            risk['state_probs']['S1_Minor']
        )
        return ll

    def sample_proposal(self, current: Dict, scale: Dict) -> Dict:
        """Generate proposal parameters with random walk."""
        proposal = {}
        proposal['h'] = max(0.1, current['h'] + np.random.normal(0, scale['h']))
        proposal['f'] = max(0.1, current['f'] + np.random.normal(0, scale['f']))
        proposal['a'] = np.clip(current['a'] + np.random.normal(0, scale['a']), 0, 1)
        proposal['s'] = int(np.clip(
            round(current['s'] + np.random.normal(0, scale['s'])), 0, 100
        ))
        return proposal

    def run(self,
            n_iter: int = 10000,
            burnin: int = 2500,
            thin:   int = 5,
            seed:   Optional[int] = None) -> pd.DataFrame:
        """
        Run Metropolis-Hastings MCMC.

        Args:
            n_iter: Total iterations
            burnin: Iterations to discard
            thin:   Keep every thin-th sample
            seed:   Random seed for reproducibility

        Returns:
            DataFrame of posterior samples
        """
        if seed is not None:
            np.random.seed(seed)

        current = {
            'h': self.initial.training,
            'f': self.initial.maint_freq,
            'a': self.initial.maint_adh,
            's': self.initial.inspection
        }
         
        scale = {'h': 5.0, 'f': 0.2, 'a': 0.1, 's': 3.0}

        current_ll   = self.log_likelihood(**current)
        current_lp   = self.log_prior(**current)
        current_post = current_ll + current_lp

        samples  = []
        accepted = 0

        for i in range(n_iter):
            prop      = self.sample_proposal(current, scale)
            prop_ll   = self.log_likelihood(**prop)
            prop_lp   = self.log_prior(**prop)
            prop_post = prop_ll + prop_lp

            if np.log(np.random.random()) < prop_post - current_post:
                current      = prop
                current_ll   = prop_ll
                current_lp   = prop_lp
                current_post = prop_post
                accepted     += 1

            if i >= burnin and i % thin == 0:
                model = BiosafetyModel(
                    training=current['h'],
                    maint_freq=current['f'],
                    maint_adh=current['a'],
                    inspection=current['s']
                )
                risk = model.risk_per_procedure()
                samples.append({
                    'training':     current['h'],
                    'maint_freq':   current['f'],
                    'maint_adh':    current['a'],
                    'inspection':   current['s'],
                    'log_risk':     risk['log_risk'],
                    'disaster_prob': risk['disaster_prob']
                })
        self.accepted = accepted  # Store as attribute
        print(f"Acceptance rate: {accepted/n_iter:.3f}")
        return pd.DataFrame(samples)


# ============================================================================
# ROI ANALYSIS
# ============================================================================

class ROIAnalysis:
    """
    Return on Investment analysis for safety interventions.

    Assumptions:
        - Disaster cost: $10 million
        - Annual procedures: 10,000
        - Intervention costs as specified
    """

    def __init__(self, base_model: BiosafetyModel):
        self.base_model        = base_model
        self.base_risk         = base_model.risk_per_procedure()
        self.disaster_cost     = 10_000_000
        self.annual_procedures = 10000

    def intervention_cost(self,
                          intervention: str,
                          current_value: float,
                          new_value: float) -> float:
        """Cost of changing intervention level."""
        if intervention == 'training':
            return abs(new_value - current_value) * 100
        elif intervention == 'maintenance_freq':
            return abs(new_value - current_value) * 10000
        elif intervention == 'maintenance_adh':
            if new_value > current_value:
                improvement = new_value - current_value
                base_cost   = 20000
                return base_cost * improvement * (1 + 2 * current_value)
            return 0
        elif intervention == 'inspection':
            if current_value < 70 <= new_value:
                return 30000
            return 0
        return 0

    def annual_benefit(self, new_model: BiosafetyModel) -> float:
        """Annual monetary benefit of risk reduction."""
        old_annual = (self.base_risk['disaster_prob']
                      * self.annual_procedures * self.disaster_cost)
        new_annual = (new_model.risk_per_procedure()['disaster_prob']
                      * self.annual_procedures * self.disaster_cost)
        return old_annual - new_annual

    def analyze_intervention(self,
                             intervention: str,
                             value_range: np.ndarray) -> pd.DataFrame:
        """Analyze ROI for a single intervention."""
        current_value = getattr(self.base_model, intervention)
        results = []
        for new_value in value_range:
            if abs(new_value - current_value) < 1e-6:
                continue
            params = {
                'training':   self.base_model.training,
                'maint_freq': self.base_model.maint_freq,
                'maint_adh':  self.base_model.maint_adh,
                'inspection': self.base_model.inspection
            }
            params[intervention] = new_value
            new_model = BiosafetyModel(**params)
            cost      = self.intervention_cost(intervention, current_value, new_value)
            benefit   = self.annual_benefit(new_model)
            roi       = benefit / cost if cost > 0 else 0
            payback   = 1 / roi if roi > 0 else np.inf
            results.append({
                'value':        new_value,
                'cost':         cost,
                'annual_benefit': benefit,
                'roi':          roi,
                'payback_years': payback,
                'log_risk':     new_model.risk_per_procedure()['log_risk']
            })
        return pd.DataFrame(results)

    def find_optimal_allocation(self, total_budget: float) -> Dict:
        """Find optimal budget allocation across interventions."""
        best_benefit    = -np.inf
        best_allocation = None

        for train_hours in [0, 20, 40, 60, 80, 100, 120]:
            for maint_freq in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
                for maint_adh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                    for insp_score in [50, 60, 70, 75, 80, 85, 90]:
                        total_cost = (
                            self.intervention_cost('training',        self.base_model.training,   train_hours) +
                            self.intervention_cost('maintenance_freq', self.base_model.maint_freq, maint_freq)  +
                            self.intervention_cost('maintenance_adh',  self.base_model.maint_adh,  maint_adh)   +
                            self.intervention_cost('inspection',       self.base_model.inspection, insp_score)
                        )
                        if total_cost > total_budget:
                            continue
                        model   = BiosafetyModel(
                            training=train_hours, maint_freq=maint_freq,
                            maint_adh=maint_adh, inspection=insp_score
                        )
                        benefit = self.annual_benefit(model)
                        if benefit > best_benefit:
                            best_benefit    = benefit
                            best_allocation = {
                                'training':   train_hours,
                                'maint_freq': maint_freq,
                                'maint_adh':  maint_adh,
                                'inspection': insp_score,
                                'cost':       total_cost,
                                'benefit':    benefit,
                                'roi':        benefit / total_cost if total_cost > 0 else 0,
                                'log_risk':   model.risk_per_procedure()['log_risk']
                            }
        return best_allocation
    
    
    def find_optimal_allocation_scenario(self, total_budget: float,
                                          n_steps: int = 10) -> Dict:
        """
        Optimal budget allocation using scenario_risk(n_steps) for evaluation.
    
        This uses multi-step absorption probability so that maintenance
        and inspection effects on intermediate-state transitions are visible.
        Clearly labeled as illustrative — not used in Bayesian updating or
        per-procedure risk reporting.
    
        Args:
            total_budget: Available budget in dollars
            n_steps:      Number of Markov steps per procedure session
    
        Returns:
            Dictionary with optimal allocation and expected benefit
        """
        # Baseline using scenario risk
        base_scenario = self.base_model.scenario_risk(n_steps=n_steps)
    
        best_benefit    = -np.inf
        best_allocation = None
    
        for train_hours in [0, 20, 40, 60, 80, 100, 120]:
            for maint_freq in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
                for maint_adh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                    for insp_score in [ 70, 75, 80, 85, 90]:
                        total_cost = (
                            self.intervention_cost('training',         self.base_model.training,   train_hours) +
                            self.intervention_cost('maintenance_freq', self.base_model.maint_freq, maint_freq)  +
                            self.intervention_cost('maintenance_adh',  self.base_model.maint_adh,  maint_adh)   +
                            self.intervention_cost('inspection',       self.base_model.inspection, insp_score)
                        )
                        if total_cost > total_budget:
                            continue
    
                        model        = BiosafetyModel(
                            training=train_hours, maint_freq=maint_freq,
                            maint_adh=maint_adh, inspection=insp_score
                        )
                        new_scenario = model.scenario_risk(n_steps=n_steps)
    
                        old_annual = (base_scenario['disaster_prob']
                                      * self.annual_procedures * self.disaster_cost)
                        new_annual = (new_scenario['disaster_prob']
                                      * self.annual_procedures * self.disaster_cost)
                        benefit    = old_annual - new_annual
    
                        if benefit > best_benefit:
                            best_benefit    = benefit
                            best_allocation = {
                                'training':   train_hours,
                                'maint_freq': maint_freq,
                                'maint_adh':  maint_adh,
                                'inspection': insp_score,
                                'cost':       total_cost,
                                'benefit':    benefit,
                                'roi':        benefit / total_cost if total_cost > 0 else 0,
                                'log_risk':   new_scenario['log_risk']
                            }
        return best_allocation


# ============================================================================
# DATA SIMULATION
# ============================================================================

def simulate_observations(model: BiosafetyModel,
                           n_months: int = 36,
                           procedures_per_month: int = 100,
                           seed: Optional[int] = None) -> pd.DataFrame:
    """
    Simulate observed incident data from a model.

    Args:
        model: True underlying model
        n_months: Number of months to simulate
        procedures_per_month: Procedures per month
        seed: Random seed

    Returns:
        DataFrame with monthly incident counts
    """
    if seed is not None:
        np.random.seed(seed)

    risk       = model.risk_per_procedure()
    p_disaster = risk['disaster_prob']
    # p_nearmiss = risk['state_probs']['S2_Serious']
    p_nearmiss = risk['state_probs']['S1_Minor']

    data = []
    for month in range(1, n_months + 1):
        incidents   = np.random.binomial(procedures_per_month, p_disaster)
        incidents= int(np.round(incidents)) 
        near_misses = np.random.binomial(procedures_per_month, p_nearmiss)
        near_misses= int(np.round(near_misses))
        data.append({
            'month':       month,
            'incidents':   incidents,
            'near_misses': near_misses
        })
    return pd.DataFrame(data)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_maintenance_heatmap():
    """Plot log-risk as function of frequency and adherence."""
    freqs = np.linspace(0.3, 2.5, 20)
    adhs  = np.linspace(0.3, 1.0, 20)
    X, Y  = np.meshgrid(freqs, adhs)
    Z     = np.zeros_like(X)

    for i in range(len(freqs)):
        for j in range(len(adhs)):
            model   = BiosafetyModel(training=50, maint_freq=freqs[i],
                                     maint_adh=adhs[j], inspection=75)
            Z[j, i] = model.risk_per_procedure()['log_risk']

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r')
    plt.colorbar(label='Log-Risk (higher = safer)')
    plt.xlabel('Maintenance Frequency (per month)')
    plt.ylabel('Maintenance Adherence')
    plt.title('Log-Risk as Function of Frequency and Adherence')
    plt.scatter([0.5, 2.0], [0.4, 0.4], c='red',   s=100, marker='x', label='Poor adherence')
    plt.scatter([0.5, 2.0], [0.9, 0.9], c='green', s=100, marker='o', label='Good adherence')
    plt.legend()
    plt.tight_layout()
    plt.savefig('maintenance_heatmap.png', dpi=150)
    plt.show()


def plot_posterior(posterior: pd.DataFrame, true_values: Dict):
    """Plot posterior distributions with true values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    params = [
        ('training',   'Training Hours',        posterior['training']),
        ('maint_freq', 'Maintenance Frequency',  posterior['maint_freq']),
        ('maint_adh',  'Maintenance Adherence',  posterior['maint_adh']),
        ('inspection', 'Inspection Score',       posterior['inspection']),
        ('log_risk',   'Log-Risk',               posterior['log_risk'])
    ]
    for idx, (name, title, data) in enumerate(params):
        ax = axes[idx // 3, idx % 3]
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(data.mean(), color='red',   linestyle='--',
                   label=f"Mean: {data.mean():.2f}")
        if name in true_values:
            ax.axvline(true_values[name], color='green', linestyle='-',
                       label=f"True: {true_values[name]:.2f}")
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Posterior: {title}')
        ax.legend(fontsize=8)
    axes[1, 2].axis('off')
    plt.suptitle('MCMC Posterior Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('posterior.png', dpi=150)
    plt.show()


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

import sys
from io import StringIO

def demonstrate():
    """Run complete demonstration of the framework."""

    output_file = "biorisk_demo.txt"
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:     
        print("\n" + "="*70)
        print("BIOSAFETY LOG-RISK MARKOV FRAMEWORK")
        print("="*70)
        print("\nA Log-Scale Markov Framework for Biosafety Risk Assessment")
        print("From WHO Classification to Dynamic Risk Management")
        print("\n" + "="*70)
    
        # [1] Prior distributions
        print("\n[1] PRIOR DISTRIBUTIONS")
        print("-" * 40)
        h_samples = np.random.lognormal(mean=np.log(50),  sigma=0.5, size=10000)
        f_samples = np.random.lognormal(mean=np.log(1.0), sigma=0.5, size=10000)
        a_samples = np.random.beta(2, 2, size=10000)
        print(f"Training hours:   median={np.median(h_samples):.1f}")
        print(f"                  95% CI=[{np.percentile(h_samples, 2.5):.1f}, {np.percentile(h_samples, 97.5):.1f}]")
        print(f"Maintenance freq: median={np.median(f_samples):.2f}")
        print(f"                  95% CI=[{np.percentile(f_samples, 2.5):.2f}, {np.percentile(f_samples, 97.5):.2f}]")
        print(f"Maintenance adh:  mean={np.mean(a_samples):.2f}")
        print(f"                  95% CI=[{np.percentile(a_samples, 2.5):.2f}, {np.percentile(a_samples, 97.5):.2f}]")
    
        # [2] Maintenance scenarios
        print("\n[2] MAINTENANCE EFFECTIVENESS: FREQUENCY vs ADHERENCE")
        print("-" * 70)
        scenarios = [
            {"name": "Low freq, poor adh",  "f": 0.5, "a": 0.4},
            {"name": "High freq, poor adh", "f": 2.0, "a": 0.4},
            {"name": "Low freq, good adh",  "f": 0.5, "a": 0.9},
            {"name": "Moderate, perfect",   "f": 1.0, "a": 1.0},
            {"name": "High freq, good adh", "f": 2.0, "a": 0.9},
        ]
        print(f"\n{'Scenario':20} {'Freq':5} {'Adh':5} {'Maint':>6} {'Log-Risk':>9} {'Disaster':>12}")
        print("-" * 70)
        for s in scenarios:
            model = BiosafetyModel(training=50, maint_freq=s['f'], maint_adh=s['a'], inspection=75)
            risk  = model.risk_per_procedure()
            print(f"{s['name']:20} {s['f']:5.1f} {s['a']:5.1f} "
                  f"{model.maint_effect:6.2f} {risk['log_risk']:9.2f} {risk['disaster_prob']:12.2e}")
    
        low_good  = BiosafetyModel(training=50, maint_freq=0.5, maint_adh=0.9, inspection=75)
        high_poor = BiosafetyModel(training=50, maint_freq=2.0, maint_adh=0.4, inspection=75)
        print("\n" + "="*70)
        print("KEY INSIGHT: Adherence > Frequency")
        print("="*70)
        print(f"\nLow frequency (0.5) with good adherence (90%):")
        print(f"  Maintenance effect: {low_good.maint_effect:.2f} logs")
        print(f"  Log-risk: {low_good.risk_per_procedure()['log_risk']:.2f}")
        print(f"\nHigh frequency (2.0) with poor adherence (40%):")
        print(f"  Maintenance effect: {high_poor.maint_effect:.2f} logs")
        print(f"  Log-risk: {high_poor.risk_per_procedure()['log_risk']:.2f}")
        print(f"\n✅ Good adherence with low frequency is "
              f"{low_good.maint_effect/high_poor.maint_effect:.1f}x more effective")
    
        # [3] MCMC
        print("\n[3] BAYESIAN UPDATING WITH SIMULATED DATA")
        print("-" * 70)
        true_model = BiosafetyModel(training=65, maint_freq=1.2, maint_adh=0.85, inspection=75)
        true_risk  = true_model.risk_per_procedure()
        observed   = simulate_observations(true_model, n_months=36, seed=123)
    
        print(f"\nTrue model (unknown to MCMC):")
        print(f"  Training:          {true_model.training:.0f} hours")
        print(f"  Maintenance freq:  {true_model.maint_freq:.2f}/month")
        print(f"  Maintenance adh:   {true_model.maint_adh:.0%}")
        print(f"  Inspection:        {true_model.inspection}")
        print(f"  Log-risk:          {true_risk['log_risk']:.2f}")
        print(f"\nSimulated 36 months of data:")
        print(f"  Total incidents:   {observed['incidents'].sum()}")
        print(f"  Total near-misses: {observed['near_misses'].sum()}")
    
        initial_model = BiosafetyModel(training=50, maint_freq=1.0, maint_adh=0.7, inspection=60)
        print(f"\nRunning MCMC (10000 iterations, burnin=2500, thin=5)...")
        mcmc      = MCMCBiosafety(initial_model, observed)
        posterior = mcmc.run(n_iter=20000, burnin=5000, thin=5, seed=456)
    
        print(f"\nPosterior summaries:")
        print(f"{'Parameter':15} {'True':>8} {'Mean':>8} {'Std':>8} {'2.5%':>8} {'97.5%':>8}")
        print("-" * 65)
        params = [
            ('training',   true_model.training,   posterior['training']),
            ('maint_freq', true_model.maint_freq, posterior['maint_freq']),
            ('maint_adh',  true_model.maint_adh,  posterior['maint_adh']),
            ('inspection', true_model.inspection, posterior['inspection']),
            ('log_risk',   true_risk['log_risk'], posterior['log_risk']),
        ]
        for name, true, post in params:
            print(f"{name:15} {true:8.2f} {post.mean():8.2f} {post.std():8.2f} "
                  f"{post.quantile(0.025):8.2f} {post.quantile(0.975):8.2f}")
    
        # [4] ROI
        print("\n[4] RETURN ON INVESTMENT ANALYSIS")
        print("-" * 70)
        updated_model = BiosafetyModel(
            training   = posterior['training'].mean(),
            maint_freq = posterior['maint_freq'].mean(),
            maint_adh  = posterior['maint_adh'].mean(),
            inspection = int(posterior['inspection'].mean())
        )
        roi = ROIAnalysis(updated_model)
    
        print(f"\nTraining ROI (from {updated_model.training:.0f} hours):")
        for hours in [20, 40, 60, 80, 100]:
            cost = roi.intervention_cost('training', updated_model.training, hours)
            if cost > 0:
                model_h = BiosafetyModel(
                    training=hours, maint_freq=updated_model.maint_freq,
                    maint_adh=updated_model.maint_adh, inspection=updated_model.inspection
                )
                benefit = roi.annual_benefit(model_h)
                roi_val = benefit / cost if cost > 0 else 0
                print(f"  → {hours:3.0f} hours: cost=${cost:,.0f}, benefit=${benefit:,.0f}/yr, ROI={roi_val:.1f}x")
    
        if updated_model.inspection < 70:
            cost    = roi.intervention_cost('inspection', updated_model.inspection, 75)
            model_i = BiosafetyModel(
                training=updated_model.training, maint_freq=updated_model.maint_freq,
                maint_adh=updated_model.maint_adh, inspection=75
            )
            benefit = roi.annual_benefit(model_i)
            roi_val = benefit / cost if cost > 0 else 0
            print(f"\nCross inspection threshold (→75): cost=${cost:,.0f}, "
                  f"benefit=${benefit:,.0f}/yr, ROI={roi_val:.1f}x")
    
        print(f"\nOptimal allocation for $100,000 budget:")
        optimal = roi.find_optimal_allocation(100000)
        if optimal:
            print(f"  Training:          {optimal['training']:.0f} hours")
            print(f"  Maintenance freq:  {optimal['maint_freq']:.2f}/month")
            print(f"  Maintenance adh:   {optimal['maint_adh']:.0%}")
            print(f"  Inspection:        {optimal['inspection']}")
            print(f"  Annual benefit:    ${optimal['benefit']:,.0f}")
            print(f"  ROI:               {optimal['roi']:.1f}x")
            print(f"  Resulting log-risk:{optimal['log_risk']:.2f}")
    
        print("\n" + "="*70)

    
        # [4b] Illustrative multi-step ROI
        print("\n[4b] ILLUSTRATIVE MULTI-STEP ROI (scenario_risk, n_steps=10)")
        print("     NOTE: for qualitative comparison only, not used in MCMC or")
        print("     per-procedure risk reporting.")
        print("-" * 70)
        print(f"\nOptimal allocation for $100,000 budget (multi-step evaluation):")
        optimal_s = roi.find_optimal_allocation_scenario(100000, n_steps=10)
        if optimal_s:
            print(f"  Training:          {optimal_s['training']:.0f} hours")
            print(f"  Maintenance freq:  {optimal_s['maint_freq']:.2f}/month")
            print(f"  Maintenance adh:   {optimal_s['maint_adh']:.0%}")
            print(f"  Inspection:        {optimal_s['inspection']}")
            print(f"  Annual benefit:    ${optimal_s['benefit']:,.0f}")
            print(f"  ROI:               {optimal_s['roi']:.1f}x")
            print(f"  Resulting log-risk:{optimal_s['log_risk']:.2f}")
    
    
        # [5] Edge class table
        print("\n[5] EDGE CLASS TABLE (manuscript appendix)")
        true_model.print_edge_classes()
        
    finally:
           # Save to UTF-8 file and restore stdout
           output_text = captured.getvalue()
           with open(output_file, 'w', encoding='utf-8') as f:
               f.write(output_text)
           sys.stdout = old_stdout
           
           print(f"✅ Complete demo saved: {output_file} ({len(output_text):,} characters)")        
    
    # Plots
    try:
        print("\nGenerating visualizations...")
        plot_maintenance_heatmap()
        true_vals = {
            'training':   true_model.training,
            'maint_freq': true_model.maint_freq,
            'maint_adh':  true_model.maint_adh,
            'inspection': true_model.inspection,
            'log_risk':   true_risk['log_risk']
        }
        plot_posterior(posterior, true_vals)
        print("✅ Plots saved: maintenance_heatmap.png, posterior.png")
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")


 
 

def test_crispr_bt_crop():
    """CRISPR Bt crop risk assessment."""
    
    output_file = "biorisk_demo.txt"
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:    
        print("="*70)
        print("CRISPR Bt CROP RISK ASSESSMENT")
        print("RG II parent → synthetic pathogenicity risk")
        print("="*70)
        
        
        
       # prior_model = BiosafetyModel(training=40, maint_freq=0.8,
       #                             maint_adh=0.6, inspection=55)
       # prior_model = BiosafetyModel.from_risk_group(2)  # Bt parent
        
        hm_rg, hm_lprior = HazardModifier.from_genetic_mods(2, mf=1.5, tp=0.2)
        prior_model = BiosafetyModel.from_risk_group(hm_rg, training=40, maint_freq=0.8,
                                                   maint_adh=0.6, inspection=55)   
    
        prior_risk = prior_model.risk_per_procedure()
        print(f"PARENT Bt (RG II prior):")
        print(f"  Log-risk: {prior_risk['log_risk']:.2f}")
        print(f"  P(disaster): {prior_risk['disaster_prob']:.2e}")
        
        synth_data = simulate_observations(prior_model, n_months=36,
                                          procedures_per_month=200, seed=789)
        synth_data.loc[12, 'near_misses'] = 3
        synth_data.loc[24, 'incidents'] = 1
        
        print(f"\nSIMULATED FIELD TRIAL DATA (36 months, 200 proc./month):")
        print(f"  Total near-misses: {synth_data['near_misses'].sum()}")
        print(f"  Total incidents:   {synth_data['incidents'].sum()}")
        
        initial_model = BiosafetyModel(training=30, maint_freq=1.0,
                                      maint_adh=0.7, inspection=60)
        mcmc = MCMCBiosafety(initial_model, synth_data)
        posterior = mcmc.run(n_iter=15000, burnin=4000, thin=5, seed=456)
        
        post_model = BiosafetyModel(
            training=posterior['training'].mean(),
            maint_freq=posterior['maint_freq'].mean(),
            maint_adh=posterior['maint_adh'].mean(),
            inspection=int(posterior['inspection'].mean())
        )
        post_risk = post_model.risk_per_procedure()
        
        print(f"\nMCMC POSTERIOR (15k iterations):")
        
        effective_samples = (15000 - 4000) / 5
        accept_rate = posterior.shape[0] / effective_samples
        
        print(f"    Acceptance rate: {accept_rate:.1%}")
        print(f"  Posterior mean log-risk: {post_risk['log_risk']:.2f}")
        print(f"  95% CrI: [{posterior['log_risk'].quantile(0.025):.2f}, "
              f"{posterior['log_risk'].quantile(0.975):.2f}]")
        print(f"  Posterior P(disaster): {post_risk['disaster_prob']:.2e}")
        
        prior_threshold = HazardModifier.T23  # 4.75 = RG2/3 boundary
        # Higher L = safer, so "below threshold" means higher risk (lower L)
        post_mean_risky = post_risk['log_risk'] < prior_threshold
        ci_upper_safe = posterior['log_risk'].quantile(0.975) >= prior_threshold
        
        print(f"\nREGULATORY IMPLICATIONS:")
        print(f"  Posterior mean {post_risk['log_risk']:.2f} "
              f"{'risky (<)' if post_mean_risky else 'safe (≥)'} "
              f"RG2/3 threshold {prior_threshold}")
        print(f"  95% CrI upper {'safe (≥)' if ci_upper_safe else 'risky (<)'} threshold")
        
        if post_mean_risky or not ci_upper_safe:
            print("  → ESCALATE to RG III containment (EU GMO Directive)")
        else:
            print("  → Remains RG II")
        
        print("="*70)
        risk_improvement = post_risk['log_risk'] - prior_risk['log_risk']
        print(f"Risk profile improvement: +{risk_improvement:+.2f} logs")

    finally:
           # Save to UTF-8 file and restore stdout
           output_text = captured.getvalue()
           with open(output_file, 'w', encoding='utf-8') as f:
               f.write(output_text)
           sys.stdout = old_stdout
           
           print(f"✅ Demo saved: {output_file} ({len(output_text):,} characters)")
    return prior_risk, post_risk, posterior           
           
        
def test_h5n1_kawaoka():
    """H5N1 Real Data (2019 UW Incident) risk assessment.
    RG III parent (ferret-transmissible), BSL3, scaled field data."""
    output_file = "biorisk_demo.txt"
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        print(70 * "=")
        print("H5N1 KAWAOKA LAB NEEDLESTICK RISK ASSESSMENT")
        print("RG III baseline, BSL-3 operations, 36mo scaled data")
        print(70 * "=")
    
        # PRIOR: RG3 H5N1 ferret-transmissible (L=4.0 target)
        rg_gof, l_gof = HazardModifier.from_genetic_mods(3, mf=1.1, tp=0.9)  # Ferret GOF 
        
        priormodel = BiosafetyModel.from_risk_group(rg_gof, training=40, maint_freq=0.8, 
                                                maint_adh=0.6, inspection=55)
        priorrisk = priormodel.risk_per_procedure()
        print(f"PARENT H5N1 RG III prior")
        print(f"Log-risk {priorrisk['log_risk']:.2f}")
        print(f"P(disaster) {priorrisk['disaster_prob']:.2e}")
    
        # FIELD TRIAL DATA: 36 months, ~200 proc/month = 7200 total
        synthdata = simulate_observations(priormodel, n_months=36, procedures_per_month=200, seed=123)
        # Scale to reported: 3 incidents, 4 near-misses (pipetting errors)
        synthdata.loc[12, 'incidents'] = 1
        synthdata.loc[24, 'incidents'] = 2  
        synthdata.loc[[6,18,30,33], 'nearmisses'] = 1
        print(f"FIELD TRIAL DATA 36 months, 200 proc/month")
        print(f"Total near-misses {synthdata['nearmisses'].sum()}")
        print(f"Total incidents {synthdata['incidents'].sum()}")
    
        # BAYESIAN UPDATE
        initialmodel = BiosafetyModel(training=30, maint_freq=1.0, maint_adh=0.7, inspection=60)
        mcmc = MCMCBiosafety(initialmodel, synthdata)
        posterior = mcmc.run(n_iter=15000, burnin=4000, thin=5, seed=456)
        
        postmodel = BiosafetyModel(
            training=posterior['training'].mean(),
            maint_freq=posterior['maint_freq'].mean(), 
            maint_adh=posterior['maint_adh'].mean(),
            inspection=int(posterior['inspection'].mean())
        )
        postrisk = postmodel.risk_per_procedure()
        
        print(f"POSTERIOR {posterior.shape[0]} effective samples")
        
        # New (compute from posterior size):
        acceptrate = posterior.shape[0] / (15000 - 4000) / 5  # niter-burnin / thin
        print(f"Acceptance rate {acceptrate:.1%}")
        print(f"Effective samples: {posterior.shape[0]}")
    
       # print(f"Acceptance rate {mcmc.accepted/mcmc.niter:.1%}")
        print(f"Posterior mean log-risk {postrisk['log_risk']:.2f}")
        print(f"95% CrI {posterior['log_risk'].quantile(0.025):.2f}, {posterior['log_risk'].quantile(0.975):.2f}")
        print(f"Posterior P(disaster) {postrisk['disaster_prob']:.2e}")
    
        # RG THRESHOLD TEST (RG3=4.75 boundary)
        priorthreshold = 4.75
        postmeanbelow = postrisk['log_risk'] > priorthreshold  # higher L = safer
        ciupperexceeds = posterior['log_risk'].quantile(0.975) < priorthreshold
        
        print("IMPLICATIONS")
        print(f"Posterior mean {postrisk['log_risk']:.2f} {'below' if not postmeanbelow else 'above'} RG III threshold {priorthreshold}")
        if not postmeanbelow or not ciupperexceeds:
            print("ESCALATE to RG IV containment (BSL4)")
        else:
            print("Remains RG III (BSL3 sufficient)")
    
        # ROI for BSL3 optimization
        roi = ROIAnalysis(postmodel)
        optimal = roi.find_optimal_allocation(100000)  # $100k budget
        
        print(f"ROI optimal allocation @ $100k budget:")
        print(f"Training {optimal['training']:.0f} hours")
        print(f"Maint freq {optimal['maint_freq']:.2f}/month") 
        print(f"Maint adh {optimal['maint_adh']:.0%}")
        print(f"Inspection {optimal['inspection']}")
        print(f"Annual benefit ${optimal['benefit']:,.0f}")
        print(f"ROI {optimal['roi']:.1f}x")
        print(f"Resulting log-risk {optimal['log_risk']:.2f}")       
        print(70 * "=")
        
        risk_improvement = optimal['log_risk'] - postrisk['log_risk']
        print(f"Risk profile improvement: +{risk_improvement:.2f} logs "
              f"(ROI {optimal['roi']:.1f}x)")
      
    finally:
           # Save to UTF-8 file and restore stdout
           output_text = captured.getvalue()
           with open(output_file, 'w', encoding='utf-8') as f:
               f.write(output_text)
           sys.stdout = old_stdout
           
           print(f"✅ Demo saved: {output_file} ({len(output_text):,} characters)")
    return priorrisk, postrisk, posterior   
           



def test_fouchier_h5n1():
    """H5N1 Real Data (2019 UW Incident) risk assessment.
    RG III parent (ferret-transmissible), BSL3, scaled field data."""
    output_file = "biorisk_demo.txt"
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    try:
        print(70 * "=")
        print("H5N1 FOUCHIER GOF (2011-12 Erasmus MC)")
        print("5 mutations → ferret airborne transmission")
        print(70 * "=")
    
        # PRIOR: RG3 H5N1 ferret-transmissible (L=4.0 target)
        rg_gof, l_gof = HazardModifier.from_genetic_mods(3, mf=1.1, tp=0.95)  # Ferret GOF 
        
        priormodel = BiosafetyModel.from_risk_group(rg_gof, training=40, maint_freq=0.8, 
                                                maint_adh=0.6, inspection=55)
        priorrisk = priormodel.risk_per_procedure()
        print(f"PARENT H5N1 RG III prior")
        print(f"Log-risk {priorrisk['log_risk']:.2f}")
        print(f"P(disaster) {priorrisk['disaster_prob']:.2e}")
    
        # Hypothetical BSL4 data (0 incidents, more procedures)
        synthdata = simulate_observations(priormodel, n_months=36, procedures_per_month=150, seed=789)
        # Lower events due to BSL4 controls
        synthdata.loc[12, 'incidents'] = 0
        synthdata.loc[24, 'nearmisses'] = 0.5  
        
        synthdata.loc[[6,18,30,33], 'nearmisses'] = 1
        print(f"FIELD TRIAL DATA 36 months, 200 proc/month")
        print(f"Total near-misses {synthdata['nearmisses'].sum()}")
        print(f"Total incidents {synthdata['incidents'].sum()}")
    
        # BAYESIAN UPDATE
        initialmodel = BiosafetyModel(training=30, maint_freq=1.0, maint_adh=0.7, inspection=60)
        mcmc = MCMCBiosafety(initialmodel, synthdata)
        posterior = mcmc.run(n_iter=15000, burnin=4000, thin=5, seed=456)
        
        postmodel = BiosafetyModel(
            training=posterior['training'].mean(),
            maint_freq=posterior['maint_freq'].mean(), 
            maint_adh=posterior['maint_adh'].mean(),
            inspection=int(posterior['inspection'].mean())
        )
        postrisk = postmodel.risk_per_procedure()
        
        print(f"POSTERIOR {posterior.shape[0]} effective samples")
        
        # New (compute from posterior size):
        acceptrate = posterior.shape[0] / (15000 - 4000) / 5  # niter-burnin / thin
        print(f"Acceptance rate {acceptrate:.1%}")
        print(f"Effective samples: {posterior.shape[0]}")
    
       # print(f"Acceptance rate {mcmc.accepted/mcmc.niter:.1%}")
        print(f"Posterior mean log-risk {postrisk['log_risk']:.2f}")
        print(f"95% CrI {posterior['log_risk'].quantile(0.025):.2f}, {posterior['log_risk'].quantile(0.975):.2f}")
        print(f"Posterior P(disaster) {postrisk['disaster_prob']:.2e}")
    
        # RG THRESHOLD TEST (RG3=4.75 boundary)
        priorthreshold = 4.75
        postmeanbelow = postrisk['log_risk'] > priorthreshold  # higher L = safer
        ciupperexceeds = posterior['log_risk'].quantile(0.975) < priorthreshold
        
        print("IMPLICATIONS")
        print(f"Posterior mean {postrisk['log_risk']:.2f} {'below' if not postmeanbelow else 'above'} RG III threshold {priorthreshold}")
        if not postmeanbelow or not ciupperexceeds:
            print("ESCALATE to RG IV containment (BSL4)")
        else:
            print("Remains RG III (BSL3 sufficient)")
        
        print(70 * "=")
        # ROI for BSL3 optimization
        roi = ROIAnalysis(postmodel)
        optimal = roi.find_optimal_allocation(100000)  # $100k budget
        
        print(f"ROI optimal allocation @ $100k budget:")
        print(f"Training {optimal['training']:.0f} hours")
        print(f"Maint freq {optimal['maint_freq']:.2f}/month") 
        print(f"Maint adh {optimal['maint_adh']:.0%}")
        print(f"Inspection {optimal['inspection']}")
        print(f"Annual benefit ${optimal['benefit']:,.0f}")
        print(f"ROI {optimal['roi']:.1f}x")
        print(f"Resulting log-risk {optimal['log_risk']:.2f}")        
        print(70 * "=")
 
        risk_improvement = optimal['log_risk'] - postrisk['log_risk']
        print(f"Risk profile improvement: +{risk_improvement:.2f} logs "
              f"(ROI {optimal['roi']:.1f}x)")

    finally:
           # Save to UTF-8 file and restore stdout
           output_text = captured.getvalue()
           with open(output_file, 'w', encoding='utf-8') as f:
               f.write(output_text)
           sys.stdout = old_stdout
           
           print(f"✅ Demo saved: {output_file} ({len(output_text):,} characters)")
    return priorrisk, postrisk, posterior

# Run demo  
if __name__ == "__main__":

    output_file = "biorisk_demo.txt"
    demos = {
        "1": ("Run demonstration", demonstrate),
        "2": ("Test H5N1 Kawaoka", test_h5n1_kawaoka),
        "3": ("Test Fouchier H5N1", test_fouchier_h5n1),
        "4": ("Test CRISPR Bt crop", test_crispr_bt_crop),
        "5": ("Run all", None),
    }

    print("\nBiorisk Framework — Select an option:")
    for key, (label, _) in demos.items():
        print(f"  {key}. {label}")

    choice = input("\nEnter choice [1-5]: ").strip()

    if choice == "5":
        for key, (label, func) in demos.items():
            if func is not None:
                print(f"\n--- {label} ---")
                func()
    elif choice in demos:
        label, func = demos[choice]
        print(f"\n--- {label} ---")
        func()
    else:
        print("Invalid choice. Please enter a number between 1 and 5.")
        
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)


 