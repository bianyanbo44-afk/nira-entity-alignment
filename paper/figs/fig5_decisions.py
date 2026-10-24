# Fig. 5 — calibrated sets become automatic merges.
# Claim: under LLM annotation noise, NIRA auto-merges about as many entities as a clean-seed oracle with a comparable
# wrong-merge share, while split CP calibrated on the same seeds sends every entity to review on ZH-EN and JA-EN (a);
# across all noise settings NIRA's auto-merge rate tracks the oracle's (b). CSLS score, singleton sets are merged.
# Bars and points are means over three training seeds x 20 calibration draws. Source data: source_data/fig5_*.csv.
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import plt, FIGW, MM, METHOD_STYLE, SPLITS, MODELS, load_runs, panel_label, save  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_data')
runs = load_runs()
METHODS = [('M1_naive', 'Split CP'), ('M4n_em_namehard', 'NIRA'), ('M0_oracle', 'Oracle')]
C_OK, C_WRONG, C_REVIEW = '#42949E', '#E28E2C', '#DCDCDC'


def dec(model, split, cond, key, method):
    rs = [r for (mo, s2, c, sd), r in runs.items() if mo == model and s2 == split and c == cond]
    a = [r['results'][key]['agg'][method] for r in rs]
    return np.mean([x['dec_merge']['mean'] for x in a]), np.mean([x['dec_wrong']['mean'] for x in a]), len(rs)


# ---- a: triage of the test entities under LLM noise ----
rows = []
for sp, sp_lab in SPLITS:
    for model in MODELS:
        for m, m_lab in METHODS:
            mg, wr, nn = dec(model, sp, 'llm', 'csls|llm|ncal3000', m)
            relevant = [r['results']['csls|llm|ncal3000']['agg'][m]['dec_merge_err']['mean']
                        for (mo, s2, c, sd), r in runs.items() if mo == model and s2 == sp and c == 'llm']
            rows.append((sp_lab, model, m_lab, nn, mg, wr, float(np.mean(relevant))))
with open(f'{SRC}/fig5_triage_llm.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['split', 'model', 'method', 'n_training_seeds', 'auto_merge_rate_all_test',
                'wrong_merge_incidence_all_test', 'error_fraction_among_merged'])
    w.writerows([list(r[:4]) + [f'{x:.5f}' for x in r[4:]] for r in rows])

fig = plt.figure(figsize=(FIGW, 104 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.19, right=0.985, bottom=0.095, top=0.915, wspace=0.42)
ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

y, ylab, ypos = [], [], 0.0
group_mid = []
for gi, (sp_lab, model) in enumerate([(s, mo) for _, s in SPLITS for mo in MODELS]):
    ys = []
    for m_lab in [m for _, m in METHODS]:
        y.append(ypos); ylab.append(m_lab); ys.append(ypos); ypos -= 1.0
    group_mid.append((np.mean(ys), f'{sp_lab}\n{model}'))
    ypos -= 0.7
y = np.array(y)
merge = np.array([r[4] for r in rows]) * 100
wrong = np.array([r[5] for r in rows]) * 100
ok = merge - wrong
ax_a.barh(y, ok, height=0.78, color=C_OK, label='Merged, correct')
ax_a.barh(y, wrong, left=ok, height=0.78, color=C_WRONG, label='Merged, wrong')
ax_a.barh(y, 100 - merge, left=merge, height=0.78, color=C_REVIEW, label='Sent to review')
ax_a.set_yticks(y); ax_a.set_yticklabels(ylab, fontsize=6)
for yi, lab in zip(y, ylab):
    if lab == 'NIRA':
        ax_a.get_yticklabels()[list(y).index(yi)].set_fontweight('bold')
for ym, lab in group_mid:
    ax_a.text(-0.24, ym, lab, transform=ax_a.get_yaxis_transform(), ha='right', va='center', fontsize=6.5)
for yi, mg, wr, lab in zip(y, merge, wrong, ylab):
    if lab == 'NIRA':
        ax_a.text(mg + 1.5, yi, f'{mg:.0f}% merged, {wr:.1f}% wrong', va='center', ha='left', fontsize=5.5,
                  color='#272727')
ax_a.set_xlim(0, 100); ax_a.set_ylim(y.min() - 0.7, y.max() + 0.7)
ax_a.set_xlabel('Test entities (%)')
ax_a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.0, columnspacing=1.2, borderaxespad=0.2)
ax_a.tick_params(axis='y', length=0)

# ---- b: auto-merge rate of NIRA vs the oracle in every setting ----
SETTINGS = [('syn0.2', 'csls|random|ncal3000', 'Random, $\\eta=0.2$', 'o'),
            ('syn0.5', 'csls|random|ncal3000', 'Random, $\\eta=0.5$', 'D'),
            ('llm', 'csls|llm|ncal3000', 'LLM annotation', 's')]
pts = []
for cond, key, lab, mk in SETTINGS:
    for model in MODELS:
        for sp, _ in SPLITS:
            o, ow, _ = dec(model, sp, cond, key, 'M0_oracle')
            n_, nw, _ = dec(model, sp, cond, key, 'M4n_em_namehard')
            pts.append((lab, model, sp, o, ow, n_, nw))
with open(f'{SRC}/fig5_merge_vs_oracle.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['noise', 'model', 'split', 'oracle_merge', 'oracle_wrong', 'nira_merge', 'nira_wrong'])
    w.writerows([list(p[:3]) + [f'{x:.5f}' for x in p[3:]] for p in pts])
ax_b.plot([0, 70], [0, 70], color='#BBBBBB', lw=0.8, zorder=0)
col = METHOD_STYLE['M4n_em_namehard']['color']
for cond, key, lab, mk in SETTINGS:
    P = np.array([(p[3], p[5]) for p in pts if p[0] == lab]) * 100
    ax_b.scatter(P[:, 0], P[:, 1], marker=mk, s=16, color=col, edgecolor='white', linewidth=0.4, label=lab, zorder=3)
ax_b.set_xlim(0, 70); ax_b.set_ylim(0, 70)
ax_b.set_xlabel('Oracle auto-merge rate (%)')
ax_b.set_ylabel('NIRA auto-merge rate (%)')
ax_b.legend(loc='upper left', handletextpad=0.2, labelspacing=0.3)

panel_label(ax_a, 'a', x_offset_pt=-66)
panel_label(ax_b, 'b', x_offset_pt=-24)
print('saved', save(fig, 'fig5_decisions'))
