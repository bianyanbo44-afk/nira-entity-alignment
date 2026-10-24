# Fig. 3 — calibration map: only NIRA reaches the target under every noise mechanism and base model.
# Each point is one (dataset, base model) setting -- three DBP15K language pairs and FB15K-DB15K, RULE and PMF: test
# coverage (x) against the mean set size relative to the clean-seed oracle of the same setting (y, log scale), means
# over three training seeds x 20 calibration draws. NACP is the published correction for uniform label noise, given
# the true noise rate or our MPE estimate. Source data: source_data/fig3_points.csv.
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, FIGW, MM, METHOD_STYLE, SPLITS_ALL as SPLITS, MODELS, load_runs, panel_label, save  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_data')
runs = load_runs()
PANELS = [('syn0.2', 'dot|random|ncal3000', 'Random noise, $\\eta=0.2$'),
          ('syn0.5', 'dot|random|ncal3000', 'Random noise, $\\eta=0.5$'),
          ('syn0.5', 'dot|hard|ncal3000', 'Look-alike noise, $\\eta=0.5$'),
          ('llm', 'dot|llm|ncal3000', 'LLM annotation noise')]
METHODS = ['M1_naive', 'M2b_rule_filter', 'B2_nacp_mpe_eta', 'B1_nacp_true_eta', 'M4_em_2view', 'M4n_em_namehard']
XMIN = 0.5

rows = []
for cond, key, title in PANELS:
    for model in MODELS:
        for sp, _ in SPLITS:
            rs = [r for (mo, s2, c, sd), r in runs.items() if mo == model and s2 == sp and c == cond]
            o = np.mean([r['results'][key]['agg']['M0_oracle']['size_mean']['mean'] for r in rs])
            for m in METHODS:
                have = [r for r in rs if m in r['results'][key]['agg']]
                if len(have) < len(rs) or not have:
                    continue  # baseline not computed for every seed of this setting yet
                cov = np.mean([r['results'][key]['agg'][m]['cov']['mean'] for r in have])
                size = np.mean([r['results'][key]['agg'][m]['size_mean']['mean'] for r in have])
                rows.append((title, model, sp, m, len(have), cov, size / o))
os.makedirs(SRC, exist_ok=True)
with open(f'{SRC}/fig3_points.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['panel', 'model', 'split', 'method', 'n_seeds', 'coverage', 'set_size_over_oracle'])
    w.writerows([list(r[:5]) + [f'{r[5]:.5f}', f'{r[6]:.5f}'] for r in rows])

fig, axes = plt.subplots(2, 2, figsize=(FIGW, 112 * MM), sharey=True, sharex=True,
                         gridspec_kw=dict(left=0.1, right=0.985, bottom=0.075, top=0.835, wspace=0.12, hspace=0.3))
axes = axes.ravel()
for ax, (cond, key, title) in zip(axes, PANELS):
    ax.axvspan(0.88, 0.92, color='#EFEFEF', lw=0, zorder=0)
    ax.axvline(0.90, color='#9E9E9E', lw=0.7, zorder=1)
    ax.axhline(1.0, color='#9E9E9E', lw=0.7, ls=(0, (2, 2)), zorder=1)
    off = []  # points left of the axis range are drawn at the edge as arrows and labelled with their coverage
    for m in METHODS:
        st = METHOD_STYLE[m]
        P = np.array([(r[5], r[6]) for r in rows if r[0] == title and r[3] == m])
        if not len(P):
            continue
        hero = m == 'M4n_em_namehard'
        inside = P[:, 0] >= XMIN
        ax.scatter(P[inside, 0], P[inside, 1], marker=st['marker'], s=(st['ms'] ** 2) * (1.15 if hero else 0.9),
                   color=st.get('mfc', st['color']), edgecolor=st['color'], linewidth=0.8, zorder=4 if hero else 3)
        for cx, cy in P[~inside]:
            ax.scatter([XMIN + 0.006], [cy], marker='<', s=14, color=st['color'], lw=0, zorder=3, clip_on=False)
            off.append((cx, cy, st['color']))
    for cx, cy, col in off:
        ax.text(XMIN + 0.016, cy, f'{cx:.2f}', fontsize=5.5, color=col, ha='left', va='center')
    ax.set_yscale('log')
    ax.set_xlim(XMIN, 1.01); ax.set_ylim(0.01, 400)
    # plain tick labels: mathtext exponents (10^k) would render below the 5-pt glyph floor
    ax.set_yticks([0.1, 1, 10, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(['0.5', '', '0.7', '', '0.9', '1.0'])
    ax.set_title(title, fontsize=7, pad=4)
for ax in axes[2:]:
    ax.set_xlabel('Test coverage')
for ax in axes[::2]:
    ax.set_ylabel('Set size / clean-seed oracle')
axes[0].text(0.872, 0.05, 'target', fontsize=5.5, color='#767676', ha='right', va='bottom')
handles = []
for m in METHODS:
    st = METHOD_STYLE[m]
    handles.append(Line2D([], [], ls='', marker=st['marker'], ms=st['ms'] * (1.07 if m == 'M4n_em_namehard' else 0.95),
                          mfc=st.get('mfc', st['color']), mec=st['color'], mew=0.8, label=st['label']))
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=3, handletextpad=0.2, columnspacing=1.4,
           frameon=False)
for ax, lab in zip(axes, 'abcd'):
    panel_label(ax, lab, x_offset_pt=-24 if lab in 'ac' else -12)
print('saved', save(fig, 'fig3_calibration_map'))
missing = [m for m in METHODS if not any(r[3] == m for r in rows)]
print('methods without data yet:', missing)
