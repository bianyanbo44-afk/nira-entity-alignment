# Fig. 6 — robustness: NIRA stays at the target for every target level, calibration-set size and noise rate, while
# split CP and NACP fail in opposite directions under the two noise mechanisms.
# Sweeps on the existing exports (pilot/scripts/robust_sweep.py): the alignment models trained with 50% random seed
# noise (DBP15K ZH/JA/FR-EN and FB15K-DB15K, RULE and PMF, three training seeds), random or look-alike noise in the
# calibration labels, 10 calibration draws per setting, inner-product score. One factor varies per column; the others
# stay at the setting of Table 1 (eta 0.5, 3000 calibration seeds, target 0.90). Top: test coverage minus the target;
# bottom: mean set size relative to the clean-seed oracle of the same draw setting. Lines: mean over the alignment
# models (geometric mean for the ratio); bands: range over the models. Source data: source_data/fig6_robustness.csv.
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, FIGW, MM, METHOD_STYLE, PILOT, panel_label, save  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_data')
METHODS = ['M1_naive', 'B1_nacp_true_eta', 'M4n_em_namehard']
NOISE = {'random': dict(ls='-', label='Random noise', filled=True),
         'hard': dict(ls=(0, (3.2, 1.6)), label='Look-alike noise', filled=False)}
ETA0, NCAL0, DELTA0 = 0.5, 3000, 0.10

models = []
for p in sorted(glob.glob(f'{PILOT}/runs/label_free_v2/robust/*_syn0.5_s*.json')):
    R = json.load(open(p, encoding='utf-8'))
    cfg = {(c['noise'], c['eta'], c['ncal']): dict(cov=np.mean([s['cov'] for s in c['splits']], 0),
                                                   size=np.mean([s['size_mean'] for s in c['splits']], 0))
           for c in R['configs']}
    models.append(dict(name=os.path.basename(p)[:-5], deltas=list(R['deltas']), methods=list(R['methods']), cfg=cfg))
assert models, 'no sweep results yet'
deltas, mnames = models[0]['deltas'], models[0]['methods']
d0, o_i = deltas.index(DELTA0), mnames.index('M0_oracle')
dl = sorted(deltas, reverse=True)  # target levels 1 - delta in increasing order
levels = [1 - d for d in dl]
etas = sorted({k[1] for k in models[0]['cfg'] if k[0] == 'random' and k[2] == NCAL0})
ncals = sorted({k[2] for k in models[0]['cfg'] if k[0] == 'random' and k[1] == ETA0})


def curve(sweep, noise, m):
    """sweep values and per-model (coverage - target, set size / oracle) arrays of shape (models, values)."""
    m_i = mnames.index(m)
    if sweep == 'level':
        vals, c = levels, (noise, ETA0, NCAL0)
        pts = [[(mo['cfg'][c]['cov'][deltas.index(d), m_i] - (1 - d),
                 mo['cfg'][c]['size'][deltas.index(d), m_i] / mo['cfg'][c]['size'][deltas.index(d), o_i])
                for d in dl] for mo in models]
    else:
        vals = ncals if sweep == 'ncal' else etas
        keys = [(noise, ETA0, nc) for nc in ncals] if sweep == 'ncal' else [(noise, e, NCAL0) for e in etas]
        pts = [[(mo['cfg'][k]['cov'][d0, m_i] - (1 - DELTA0), mo['cfg'][k]['size'][d0, m_i] / mo['cfg'][k]['size'][d0, o_i])
                for k in keys] for mo in models]
    P = np.array(pts)
    return vals, P[:, :, 0], P[:, :, 1]


SWEEPS = [('level', 'Target coverage level', 'Target coverage $1-\\delta$', lambda v: f'{v:.2f}'),
          ('ncal', 'Calibration-set size', 'Calibration seeds $n$', lambda v: f'{v:g}'),
          ('eta', 'Calibration-noise rate', 'Noise rate $\\eta$', lambda v: f'{v:g}')]
rows = []
for sweep, _, _, fmt in SWEEPS:
    for noise in NOISE:
        for m in METHODS:
            vals, gap, ratio = curve(sweep, noise, m)
            for j, v in enumerate(vals):
                rows.append([sweep, noise, fmt(v), m, len(models), f'{gap[:, j].mean():.5f}', f'{gap[:, j].min():.5f}',
                             f'{gap[:, j].max():.5f}', f'{np.exp(np.log(ratio[:, j]).mean()):.5f}',
                             f'{ratio[:, j].min():.5f}', f'{ratio[:, j].max():.5f}'])
os.makedirs(SRC, exist_ok=True)
with open(f'{SRC}/fig6_robustness.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['sweep', 'noise', 'value', 'method', 'n_models', 'coverage_minus_target_mean', 'coverage_minus_target_min',
                'coverage_minus_target_max', 'size_over_oracle_geomean', 'size_over_oracle_min', 'size_over_oracle_max'])
    w.writerows(rows)

fig, axes = plt.subplots(2, 3, figsize=(FIGW, 92 * MM), sharex='col',
                         gridspec_kw=dict(left=0.088, right=0.99, bottom=0.105, top=0.87, wspace=0.3, hspace=0.12))
lo_r = min(float(r[9]) for r in rows); hi_r = max(float(r[10]) for r in rows)
for col, (sweep, title, xlabel, fmt) in enumerate(SWEEPS):
    ac, ar = axes[0, col], axes[1, col]
    ac.axhspan(-0.02, 0.02, color='#EAEAEA', lw=0, zorder=0)
    ac.axhline(0, color='#9E9E9E', lw=0.7, zorder=1)
    ar.axhline(1.0, color='#9E9E9E', lw=0.7, ls=(0, (2, 2)), zorder=1)
    for noise, ns in NOISE.items():
        for m in METHODS:
            st, hero = METHOD_STYLE[m], m == 'M4n_em_namehard'
            vals, gap, ratio = curve(sweep, noise, m)
            x = np.arange(len(vals))
            for ax, Y, mid in ((ac, gap, gap.mean(0)), (ar, ratio, np.exp(np.log(ratio).mean(0)))):
                ax.fill_between(x, Y.min(0), Y.max(0), color=st['color'], alpha=0.22 if hero else 0.09, lw=0,
                                zorder=3 if hero else 2)
                ax.plot(x, mid, color=st['color'], ls=ns['ls'], lw=1.3 if hero else 0.9, marker=st['marker'],
                        ms=st['ms'] * 0.62, mfc=st['color'] if ns['filled'] else 'white', mec=st['color'], mew=0.7,
                        zorder=5 if hero else 4)
    vals = levels if sweep == 'level' else (ncals if sweep == 'ncal' else etas)
    x0 = [round(v, 6) for v in vals].index(round({'level': 1 - DELTA0, 'ncal': NCAL0, 'eta': ETA0}[sweep], 6))
    for ax in (ac, ar):  # the setting of the main tables
        ax.axvline(x0, color='#C8C8C8', lw=0.5, ls=(0, (1, 1.5)), zorder=0)
    ar.set_xlim(-0.35, len(vals) - 0.65)
    ar.set_xticks(np.arange(len(vals))); ar.set_xticklabels([fmt(v) for v in vals])
    ar.set_yscale('log')
    ar.set_ylim(10 ** np.floor(np.log10(lo_r) - 0.05), 10 ** np.ceil(np.log10(hi_r) + 0.05))
    dec = np.arange(np.floor(np.log10(lo_r) - 0.05), np.ceil(np.log10(hi_r) + 0.05) + 1)
    ar.set_yticks(10 ** dec)
    ar.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))  # plain labels; 10^k glyphs fall below 5 pt
    ar.yaxis.set_minor_locator(NullLocator())
    ac.set_ylim(-0.7, 0.3)
    ac.set_yticks([-0.6, -0.4, -0.2, 0, 0.2])
    ac.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.1f}' if abs(v) > 1e-9 else '0'))
    ac.set_title(title, fontsize=7, pad=4)
    ar.set_xlabel(xlabel)
axes[0, 0].set_ylabel('Test coverage $-$ target')
axes[1, 0].set_ylabel('Set size / clean-seed oracle')
handles = [Line2D([], [], color=METHOD_STYLE[m]['color'], lw=1.3 if m == 'M4n_em_namehard' else 0.9,
                  marker=METHOD_STYLE[m]['marker'], ms=METHOD_STYLE[m]['ms'] * 0.62, label=METHOD_STYLE[m]['label'])
           for m in METHODS]
handles += [Line2D([], [], color='#4D4D4D', lw=0.9, ls=ns['ls'], marker='o', ms=2.6,
                   mfc='#4D4D4D' if ns['filled'] else 'white', mec='#4D4D4D', mew=0.7, label=ns['label'])
            for ns in NOISE.values()]
handles.append(Patch(facecolor='#EAEAEA', edgecolor='none', label='Target $\\pm$0.02'))
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=6, handlelength=2.3,
           handletextpad=0.35, columnspacing=1.0, frameon=False)
for col, lab in enumerate('abc'):
    panel_label(axes[0, col], lab, x_offset_pt=-24 if col == 0 else -20)
print(f'{len(models)} alignment models:', ', '.join(mo['name'] for mo in models))
print('saved', save(fig, 'fig6_robustness'))
