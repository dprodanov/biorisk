# test_biorisk.py
import pytest
from biorisk import compute_mfpt, LogRiskScale

def test_compute_mfpt():
    Q = [[0.9, 0.1, 0], [0, 0.8, 0.2], [0, 0, 1]]  # absorbing chain
    mfpt = compute_mfpt(Q)
    assert len(mfpt) == 2  # non-absorbing states
    assert mfpt[0] > 0

def test_logrisk_scale():
    scale = LogRiskScale()
    risk_db = scale.to_decibels(1e-6)
    assert -60 <= risk_db <= -50  # 1-in-million = -60 dB
