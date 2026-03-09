# BioRisk: A Bayesian Framework for Laboratory Biorisk Quantification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## Overview

This repository contains the Python implementation of a quantitative
Bayesian framework for laboratory biosafety risk management, as described in:

> D. Prodanov, *Making Biorisk Measurable: A Bayesian Framework for
> Laboratory Risk Management*, preprint, 2026.

The framework combines WHO Risk Group classification with a five-state
Markov chain model of the incident–disaster escalation chain, reporting
risk on a **log-risk scale** analogous to the decibel scale. It supports
Bayesian updating from local incident data and provides return-on-investment
analysis for three intervention types: training, preventive maintenance,
and compliance inspection.

---

## Key Features

- **Log-risk metric** — converts disaster probabilities into an additive
  scale where higher values mean safer conditions
- **Five-state Markov chain** — models escalation from normal operation
  to laboratory-acquired infection
- **Three intervention channels** — training (Hill function), maintenance
  (frequency × adherence with gap penalty), inspection (threshold effect)
- **Bayesian MCMC updating** — Metropolis–Hastings sampler with
  three-observable likelihood (near-misses, light incidents, disasters)
- **ROI and budget optimisation** — monetary and dimensionless
  cost-effectiveness metrics across all intervention types
- **ISO 35001:2019 and WHO LBM4 alignment** — framework maps directly
  onto standard biorisk management requirements


---
## Requirements

numpy

scipy

pandas

typing

---

## Installation

```bash
git clone https://github.com/dprodanov/biorisk.git
cd biorisk
pip install -r requirements.txt
```

## Example use  

```python
from biosafety_model_v54 import (
    BiosafetyModel,
    MCMCBiosafety,
    simulate_observations,
    training_roi,
    maintenance_roi_log,
    inspection_roi_log,
    optimal_allocation,
)

# --- Instantiate a BSL-3 facility model ---
model = BiosafetyModel(
    training=50.0,       # hours
    maint_freq=1.0,      # sessions per month
    maint_adh=0.8,       # fraction of sessions completed
    inspection=75.0,     # score 0-100 (threshold at 70)
    natech_rate=7.5,     # site profile (Brussels inland default)
)

# --- Compute the three log-risk metrics ---
mt = model.metrics(lambda_proc=500)  # 500 procedures/month
print(f"L       = {mt['L']:.3f}")        # training channel
print(f"logM_S1 = {mt['logM_S1']:.3f}") # maintenance channel
print(f"logM_S2 = {mt['logM_S2']:.3f}") # inspection channel

# --- Simulate 36 months of observations ---
data = simulate_observations(model, n_months=36,
                             procedures_per_month=500, seed=42)
print(data[["near_misses", "light_incidents", "disasters"]].sum())

# --- Bayesian update from observed data ---
mcmc = MCMCBiosafety(initial_model=model, observed_data=data)
posterior = mcmc.run(n_iter=20000, burnin=5000, thin=5, seed=99)
print(posterior[["L", "logM_S1", "logM_S2"]].describe())

# --- ROI analysis ---
base = BiosafetyModel(training=20, maint_freq=0.5,
                      maint_adh=0.5, inspection=50)

r_train = training_roi(base, new_training=60,
                       N_annual=10_000, V=50_000_000)
print(f"Training ROI: {r_train['roi']:.1f}x, "
      f"payback {r_train['payback_yr']:.2f} yr")

r_maint = maintenance_roi_log(base, new_freq=1.0, new_adh=0.9)
print(f"Maintenance: +{r_maint['delta_logM_S1']:.3f} logs, "
      f"{r_maint['log_per_1k']:.4f} DlogM/$1k")

r_insp = inspection_roi_log(base, new_inspection=75)
print(f"Inspection: +{r_insp['delta_logM_S2']:.3f} logs, "
      f"{r_insp['log_per_1k']:.4f} DlogM/$1k")

# --- Optimal $100k budget allocation ---
best = optimal_allocation(base, budget=100_000,
                          N_annual=10_000, V=50_000_000)
print(f"Optimal: training={best['training']:.0f}h, "
      f"freq={best['maint_freq']:.1f}, adh={best['maint_adh']:.0%}, "
      f"insp={best['inspection']:.0f}")
print(f"Cost: ${best['cost']:,}, benefit: ${best['total_benefit']:,.0f}/yr")

# --- Run full demonstration (reproduces all paper tables) ---
# python biosafety_model_v54.py
```

---
## Mathematical Specification

A complete self-contained mathematical specification of the framework
is provided in paper/biorisk_framework_spec.tex (Supplementary
Material S1 of the manuscript). It covers:

- Log-risk metric and cMFPT definition

- Five-state Markov chain and edge class taxonomy

- All three intervention effect functions with parameter justification

- Bayesian likelihood, prior distributions, and MCMC implementation

- Cost-effectiveness metrics and assumed intervention costs

- Full intervention weight matrix with design rationale