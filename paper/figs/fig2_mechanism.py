# Fig. 2 — why the name-hard kernel identifies LLM annotation noise (one calibration draw, RULE, ZH-EN, seed 3408).
# Claim: wrong LLM labels sit where only the name-hard kernel puts mass in the two-view plane (a); without that kernel EM
# treats them as correct (b), and only the full model moves the calibration threshold to where clean seeds put it (c).
# Source data: source_data/fig2_seeds.csv, fig2_kernels.csv, fig2_summary.json (written by fig2_data.py).
import json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, FIGW, MM, METHOD_STYLE, panel_label, save  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_data')
d = np.loadtxt(f'{SRC}/fig2_seeds.csv', delimiter=',', skiprows=1)
s, v, wrong, p_wo, p_nira, s_true = d.T
wrong = wrong.astype(bool)
k = np.loadtxt(f'{SRC}/fig2_kernels.csv', delimiter=',', skiprows=1)
summ = json.load(open(f'{SRC}/fig2_summary.json'))

C_CORRECT, C_WRONG = '#3775BA', '#E28E2C'
KERNELS = [(0, 'Random', '#8C8C8C'), (1, 'Structure-hard', '#42949E'), (2, 'Name-hard', '#9C4A0B')]
C_WO, C_NIRA = METHOD_STYLE['M4_em_2view']['color'], METHOD_STYLE['M4n_em_namehard']['color']
XLIM, YLIM = (-0.55, 2.1), (-0.03, 1.04)

fig = plt.figure(figsize=(FIGW, 84 * MM))
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], left=0.095, right=0.985, bottom=0.105, top=0.885, wspace=0.42, hspace=0.72)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

# ---- a: two-view plane with the label-free kernels ----
ax_a.scatter(s[~wrong], v[~wrong], s=1.6, color=C_CORRECT, alpha=0.35, lw=0, rasterized=True,
             label=f'Correct seed ({(~wrong).mean():.0%})')
ax_a.scatter(s[wrong], v[wrong], s=2.4, color=C_WRONG, alpha=0.75, lw=0, rasterized=True,
             label=f'Wrong LLM label ({wrong.mean():.0%})')
for kid, name, col in KERNELS:
    m = k[:, 2] == kid
    H, xe, ye = np.histogram2d(k[m, 0], k[m, 1], bins=[140, 140], range=[XLIM, YLIM])
    H = gaussian_filter(H, 2.0)
    H /= H.sum()
    hs = np.sort(H.ravel())[::-1]
    level = hs[np.searchsorted(np.cumsum(hs), 0.75)]  # contour enclosing 75% of the kernel's mass
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    ax_a.contour(xc, yc, H.T, levels=[level], colors=[col], linewidths=1.1)
LABEL_POS = {0: (-0.53, 0.12), 1: (1.30, 0.30), 2: (-0.53, 0.93)}
for kid, name, col in KERNELS:
    x0, y0 = LABEL_POS[kid]
    ax_a.text(x0, y0, f'{name}\nkernel', color=col, fontsize=6.5, fontweight='bold', ha='left', va='center')
ax_a.set_xlim(*XLIM); ax_a.set_ylim(*YLIM)
ax_a.set_xlabel('Alignment score $s$ (RULE embeddings)')
ax_a.set_ylabel('Name similarity $v$ (held-out view)')
leg = ax_a.legend(loc='lower left', bbox_to_anchor=(0.06, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0,
                  markerscale=3.0, borderaxespad=0.2)
for h in leg.legend_handles:
    h.set_alpha(1.0)

# ---- b: posterior that a wrong seed is correct, without and with the name-hard kernel ----
bins = np.linspace(0, 1, 11)
centers, width = (bins[:-1] + bins[1:]) / 2, 0.042
h_wo = np.histogram(p_wo[wrong], bins=bins)[0] / wrong.sum() * 100
h_ni = np.histogram(p_nira[wrong], bins=bins)[0] / wrong.sum() * 100
ax_b.bar(centers - width / 2, h_wo, width=width, color=C_WO, label='Without name-hard kernel')
ax_b.bar(centers + width / 2, h_ni, width=width, color=C_NIRA, label='NIRA')
w_wo = p_wo[wrong].sum() / p_wo.sum(); w_ni = p_nira[wrong].sum() / p_nira.sum()
ax_b.text(0.5, 99, f'Weight on wrong labels:\n{w_wo:.0%}  →  {w_ni:.1%}', ha='center', va='top', fontsize=6)
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 105)
ax_b.set_xlabel('Posterior that the label is correct')
ax_b.set_ylabel('Wrong labels (%)')
ax_b.legend(loc='center', bbox_to_anchor=(0.5, 0.45), ncol=1, handlelength=1.0, labelspacing=0.3)

# ---- c: weighted lower tail of the calibration scores and the resulting thresholds ----
def wcdf(x, w):
    o = np.argsort(x); x, w = x[o], w[o]
    return x, np.cumsum(w) / w.sum()


def wq(x, w, q):
    xs, c = wcdf(x, w)
    return xs[np.searchsorted(c, q)]


ones = np.ones_like(s)
C_YMAX = 0.42
curves = [('Clean-seed oracle', s_true, ones, '#272727', (0, (1.2, 1.2)), 1.1),
          ('Split CP', s, ones, METHOD_STYLE['M1_naive']['color'], '-', 1.1),
          ('Without name-hard kernel', s, p_wo, C_WO, (0, (3, 1.5)), 1.3),
          ('NIRA', s, p_nira, C_NIRA, '-', 1.3)]
for lab, x, w, col, ls, lw in curves:
    xs, c = wcdf(x, w)
    keep = c <= C_YMAX  # draw only the visible lower tail (no clipped path segments outside the panel)
    ax_c.plot(xs[keep], c[keep], color=col, ls=ls, lw=lw, label=lab)
ax_c.axhline(0.10, color='#BBBBBB', lw=0.6, zorder=0)
ax_c.text(1.02, 0.112, '$\\delta = 0.10$', fontsize=6, color='#767676', ha='right', va='bottom')
for lab, x, w, col, ls, lw in curves:
    t = wq(x, w, 0.10)
    ax_c.plot([t, t], [0, 0.10], color=col, lw=0.8, ls=':')
ax_c.set_xlim(-0.3, 1.05); ax_c.set_ylim(0, C_YMAX)
ax_c.set_xlabel('Alignment score $s$ of the calibration label')
ax_c.set_ylabel('Weighted CDF')
# the curves coincide in two pairs; label each pair once, next to it
ax_c.text(-0.28, 0.405, 'Split CP and\nw/o name-hard', color=METHOD_STYLE['M1_naive']['color'],
          fontsize=6, ha='left', va='top')
ax_c.text(0.48, 0.035, 'NIRA and oracle', color=C_NIRA, fontsize=6, ha='left', va='center')

for ax, lab in [(ax_a, 'a'), (ax_b, 'b'), (ax_c, 'c')]:
    panel_label(ax, lab, x_offset_pt=-22)
pdf = save(fig, 'fig2_mechanism')
print('saved', pdf)
print(f"thresholds (10% weighted quantile of s): oracle {wq(s_true, ones, .1):.3f} | split {wq(s, ones, .1):.3f} | "
      f"w/o {wq(s, p_wo, .1):.3f} | NIRA {wq(s, p_nira, .1):.3f}")
