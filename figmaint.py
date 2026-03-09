# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:57:43 2026

@author: dimit
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ── Data ──────────────────────────────────────────────────────────────────────
train_h   = [40, 60, 80, 100, 120]
train_ben = [5.7, 10.5, 13.3, 14.9, 15.8]   # annual benefit $k/yr
roi_labels = ["2.9×", "2.6×", "2.2×", "1.9×", "1.6×"]

maint_labels = ["f=0.5\na=0.9", "f=1.0\na=0.8", "f=1.5\na=0.8",
                "f=2.0\na=0.9", "f=2.0\na=0.95"]
maint_dlogm  = [0.142, 0.199, 0.260, 0.411, 0.472]   # ΔL_M log units
eff_labels   = ["0.0089", "0.0117", "0.0118", "0.0133", "0.0143"]  # per $1k

# ── Figure ────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "(A) Training: annual benefit vs. hours",
        "(B) Maintenance: \u0394L\u2098 vs. scenario",
    ],
    horizontal_spacing=0.16,
)

# Panel A — training
fig.add_trace(go.Bar(
    x=[f"{h}h" for h in train_h],
    y=train_ben,
    text=[f"${b:.1f}k\nROI {r}" for b, r in zip(train_ben, roi_labels)],
    textposition="outside",
    marker_color="#4C72B0",
    showlegend=False,
), row=1, col=1)

# Panel B — maintenance (one colour per scenario)
fig.add_trace(go.Bar(
    x=maint_labels,
    y=maint_dlogm,
    text=[f"+{v:.3f}\n{e}/$1k" for v, e in zip(maint_dlogm, eff_labels)],
    textposition="outside",
    marker_color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
    showlegend=False,
), row=1, col=2)

# ── Axes ──────────────────────────────────────────────────────────────────────
fig.update_xaxes(title_text="Training hours",            row=1, col=1)
fig.update_xaxes(title_text="Scenario (freq, adherence)",row=1, col=2,
                 tickangle=0)
fig.update_yaxes(title_text="Annual benefit ($k/yr)",    row=1, col=1,
                 range=[0, 24])
fig.update_yaxes(title_text="\u0394L\u2098 (log units)", row=1, col=2,
                 range=[0, 0.65])

# ── Global layout ─────────────────────────────────────────────────────────────
fig.update_layout(
    # title=dict(
    #     text="ROI of Training and Maintenance Interventions",
    #     x=0.5, xanchor="center", font=dict(size=19),
    # ),
    width=1000, height=480,
    margin=dict(t=100, b=80, l=70, r=40),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=11),
)

# Subplot title font size
for ann in fig.layout.annotations:
    ann.font.size = 14

fig.update_xaxes(showline=True, linecolor="black", mirror=True)
fig.update_yaxes(showline=True, linecolor="black", mirror=True,
                 showgrid=True, gridcolor="#EEEEEE")
fig.update_traces(cliponaxis=False)

# ── Export ────────────────────────────────────────────────────────────────────
fig.write_image("roi_combined.png", scale=2)

with open("roi_combined.png.meta.json", "w") as f:
    json.dump({
        "caption": (
            "ROI of training (a) and maintenance (b) interventions. "
            "Bar labels show absolute gain and cost-efficiency per $1,000."
        ),
        "description": (
            "Panel (a): bars show annual monetary benefit ($k/yr) for five "
            "training levels (40-120h above baseline 20h); bar labels include "
            "ROI multiplier. "
            "Panel (b): bars show DeltaLogM improvement for five maintenance "
            "scenarios; bar labels include cost-efficiency (DeltaLogM per $1k). "
            "Baseline: training=20h, freq=0.5/mo, adh=50%, insp=50, "
            "V=$50M, N_proc=10,000/yr."
        ),
    }, f)

print("Saved: roi_combined.png")
