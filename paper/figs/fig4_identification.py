# Fig. 4 — NIRA identifies how much of the seed set is wrong and why.
# Claim: across base models, datasets (DBP15K language pairs, FB15K-DB15K) and noise mechanisms, NIRA's noise-rate
# estimate follows the realised rate while the single-view estimate and the model without the name-hard kernel do not
# (a), and the fitted kernel weights name the mechanism that produced the noise (b). Points are means over three
# training seeds; each run averages 20 calibration draws. Source data: source_data/fig4_eta.csv, fig4_beta.csv.
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, FIGW, MM, METHOD_STYLE, SPLITS_ALL as SPLITS, MODELS, load_runs, panel_label, save  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_data')
runs = load_runs()
NOISE = [('syn0.2', 'dot|random|ncal3000', 'Random, $\\eta=0.2$', 'o'),
         ('syn0.5', 'dot|random|ncal3000', 'Random, $\\eta=0.5$', 'o'),
         ('syn0.5', 'dot|hard|ncal3000', 'Look-alike', '^'),
         ('llm', 'dot|llm|ncal3000', 'LLM annotation', 's')]
EST = [('M5_mpe_deconv', 'Single-view MPE', '#7884B4'), ('M4_em_2view', 'NIRA w/o name-hard kernel', '#E9A6A1'),
       ('M4n_em_namehard', 'NIRA (ours)', '#B64342')]

rows_eta, rows_beta = [], []
for cond, key, nlab, mk in NOISE:
    for model in MODELS:
        for sp, _ in SPLITS:
            rs = [r for (mo, s2, c, sd), r in runs.items() if mo == model and s2 == sp and c == cond]
            true = np.mean([r['results'][key]['agg']['_true_eta']['mean'] for r in rs])
            est = [np.mean([r['results'][key]['agg'][m]['eta_hat']['mean'] for r in rs]) for m, _, _ in EST]
            rows_eta.append((nlab, model, sp, len(rs), true, *est))
        beta = np.mean([[r['results'][key]['agg']['M4n_em_namehard'][b]['mean'] for b in ('beta_random', 'beta_shard', 'beta_vhard')]
                        for (mo, s2, c, sd), r in runs.items() if mo == model and c == cond], 0)
        rows_beta.append((nlab, model, *beta))
os.makedirs(SRC, exist_ok=True)
with open(f'{SRC}/fig4_eta.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['noise', 'model', 'split', 'n_seeds', 'true_eta', 'eta_single_view_mpe', 'eta_nira_wo_namehard', 'eta_nira'])
    w.writerows([list(r[:4]) + [f'{x:.5f}' for x in r[4:]] for r in rows_eta])
with open(f'{SRC}/fig4_beta.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['noise', 'model', 'beta_random', 'beta_structure_hard', 'beta_name_hard'])
    w.writerows([list(r[:2]) + [f'{x:.4f}' for x in r[2:]] for r in rows_beta])

fig = plt.figure(figsize=(FIGW, 70 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.1, right=0.985, bottom=0.21, top=0.85, wspace=0.36)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

# ---- a: estimated vs realised noise rate ----
ax_a.plot([0, 0.65], [0, 0.65], color='#BBBBBB', lw=0.8, zorder=0)
for j, (m, lab, col) in enumerate(EST):
    for nlab_, _, _, mk in [(n[2], *n[:2], n[3]) for n in NOISE]:
        pts = np.array([(r[4], r[5 + j]) for r in rows_eta if r[0] == nlab_])
        hero = m == 'M4n_em_namehard'
        ax_a.scatter(pts[:, 0], pts[:, 1], marker=mk, s=16 if hero else 12, color=col if hero else 'white',
                     edgecolor=col, linewidth=0.9, zorder=3 if hero else 2)
ax_a.set_xlim(-0.02, 0.65); ax_a.set_ylim(-0.02, 0.65)
ax_a.set_xticks([0, 0.2, 0.4, 0.6]); ax_a.set_yticks([0, 0.2, 0.4, 0.6])
ax_a.set_xlabel('Realised share of wrong calibration labels')
ax_a.set_ylabel('Estimated noise rate $\\hat\\eta$')
# legend: estimators by colour, noise mechanisms by marker shape
from matplotlib.lines import Line2D
h_est = [Line2D([], [], ls='', marker='o', ms=4, mfc=(c if m == 'M4n_em_namehard' else 'white'), mec=c, mew=0.9, label=l)
         for m, l, c in EST]
h_noise = [Line2D([], [], ls='', marker=mk, ms=4, mfc='#767676', mec='#767676', label=lab) for _, _, lab, mk in NOISE[1:]]
h_noise[0].set_label('Random')
leg1 = ax_a.legend(handles=h_est, loc='upper left', handletextpad=0.2, labelspacing=0.3, borderaxespad=0.1)
ax_a.add_artist(leg1)
ax_a.legend(handles=h_noise, loc='upper left', bbox_to_anchor=(0.0, 0.70), handletextpad=0.2, labelspacing=0.3,
            borderaxespad=0.1)

# ---- b: fitted kernel weights (share of the estimated noise mass) ----
KCOL = ['#8C8C8C', '#42949E', '#9C4A0B']
KLAB = ['Random', 'Structure-hard', 'Name-hard']
groups = [n[2] for n in NOISE]
GROUP_TXT = ['Random\n$\\eta=0.2$', 'Random\n$\\eta=0.5$', 'Look-alike', 'LLM']
x, labels = [], []
pos = 0.0
for g in groups:
    for model in MODELS:
        x.append(pos); labels.append(model); pos += 0.9
    pos += 0.6
x = np.array(x)
B = np.array([[r[2], r[3], r[4]] for g in groups for model in MODELS for r in rows_beta if r[0] == g and r[1] == model])
bottom = np.zeros(len(x))
for kk in range(3):
    ax_b.bar(x, B[:, kk] * 100, bottom=bottom, width=0.75, color=KCOL[kk], label=KLAB[kk], edgecolor='white', lw=0.4)
    bottom += B[:, kk] * 100
ax_b.set_xticks(x); ax_b.set_xticklabels(labels, fontsize=6)
for i, g in enumerate(groups):
    ax_b.text((x[2 * i] + x[2 * i + 1]) / 2, -12, GROUP_TXT[i], ha='center', va='top', fontsize=6.5)
ax_b.set_ylim(0, 100); ax_b.set_xlim(x[0] - 0.6, x[-1] + 0.6)
ax_b.set_ylabel('Share of estimated noise (%)')
ax_b.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.0, columnspacing=0.9, borderaxespad=0.2,
            title='Kernel:', title_fontsize=6, alignment='left')
ax_b.tick_params(axis='x', length=0, pad=2)

for ax, lab in [(ax_a, 'a'), (ax_b, 'b')]:
    panel_label(ax, lab, x_offset_pt=-24)
print('saved', save(fig, 'fig4_identification'))
