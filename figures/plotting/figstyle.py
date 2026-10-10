"""Shared style for the revised EG-CPW figure set (Figure 2-7).

MDPI Sensors oriented: single/double column sizes, fonts >= 7 pt,
colorblind-safe Okabe-Ito palette, PDF (vector) + PNG (600 dpi) output
into figures/final/.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

FINAL_DIR = Path(__file__).resolve().parents[2] / 'figures' / 'final'
FINAL_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR = Path(__file__).resolve().parents[2] / 'supplementary_figures'
SUPP_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
C_C0 = '#D55E00'   # vermillion  (baseline)
C_S0 = '#0072B2'   # blue        (peak weight)
C_GT = '#009E73'   # bluish green
C_ORANGE = '#E69F00'
C_SKY = '#56B4E9'
C_PINK = '#CC79A7'
C_GRAY = '#7F7F7F'

GROUP_COLORS = {'Low': C_ORANGE, 'Mid': C_SKY, 'High': C_GT}

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.7,
    'lines.linewidth': 1.4,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.35,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian',
               'traffic_cone']


def save(fig, name, supplementary=False):
    """Save PDF (vector) + PNG (600 dpi) into figures/final/ (or
    supplementary_figures/ when supplementary=True)."""
    out = SUPP_DIR if supplementary else FINAL_DIR
    pdf = out / f'{name}.pdf'
    png = out / f'{name}.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    print(f'saved {out.name}/{pdf.name} / {png.name}')
