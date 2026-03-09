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
matplotlib
plotly
kaleido

---

## Installation

```bash
git clone https://github.com/dprodanov/biorisk.git
cd biorisk
pip install -r requirements.txt
 
## Example use 

from biosafety_model_v54 import BiosafetyModel

# Initialise with intervention parameters
model = BiosafetyModel(
    training_hours=50,
    maint_frequency=1.0,   # sessions per month
    maint_adherence=0.9,   # fraction completed
    inspection_score=75,   # 0-100
)

# Compute log-risk metrics
print(f"L   = {model.L:.3f}")    # training channel
print(f"L_M = {model.L_M:.3f}") # maintenance channel
print(f"L_I = {model.L_I:.3f}") # inspection channel

# Run Bayesian update with local incident data
posterior = model.mcmc_update(
    n_procedures=18000,
    near_misses=6,
    light_incidents=17,
    disasters=0,
    n_iter=20000,
)
print(posterior.summary())

# ROI analysis
roi = model.compute_roi(budget=100_000)
print(roi.optimal_allocation())
```bash

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