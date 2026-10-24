# Generate the LaTeX result tables of the paper directly from the run files (no hand-copied numbers).
# usage: python -I make_tables.py   (run from anywhere; paths are absolute)
import glob, json, os, re
from pathlib import Path
import numpy as np

PILOT = str(Path(__file__).resolve().parents[1] / 'pilot')
OUT = str(Path(__file__).resolve().parent / 'tables')
os.makedirs(OUT, exist_ok=True)
# res_b / res_pmf_b: the same analyses re-run with the NACP baselines (identical draws; earlier methods unchanged)
CORRECTED = f'{PILOT}/runs/label_free_v2'
SRC = {('ALL', model): [f'{CORRECTED}/main/{model.lower()}'] for model in ('RULE', 'PMF')}
runs = {}
for (ds, model), dirs in SRC.items():
    for d in dirs:
        for p in glob.glob(os.path.join(d, '*_results.json')):
            m = re.match(r'(zh_en|ja_en|fr_en|fbdb)_(syn0\.2|syn0\.5|llm)_s(\d+)_results\.json', os.path.basename(p))
            if m and (model, m.group(1), m.group(2), int(m.group(3))) not in runs:
                runs[(model, m.group(1), m.group(2), int(m.group(3)))] = json.load(open(p, encoding='utf-8'))

assert len(runs) == 72, f'Expected 72 corrected main results, found {len(runs)}'
for mode in ('robust', 'hparam'):
    assert len(glob.glob(f'{CORRECTED}/{mode}/*.json')) == 48, f'Incomplete {mode} reanalysis'

SPLITS = [('zh_en', 'ZH--EN'), ('ja_en', 'JA--EN'), ('fr_en', 'FR--EN')]
MODELS = ['RULE', 'PMF']
METHODS = [('M0_oracle', 'Clean-seed oracle$^{\\dagger}$'), ('M1_naive', 'Split CP'),
           ('M2_rule_weight', 'Rel.-weighted CP'), ('M2b_rule_filter', 'Rel.-filtered CP'),
           ('B1_nacp_true_eta', 'NACP, true $\\eta^{\\ddagger}$'), ('B2_nacp_mpe_eta', 'NACP, estimated $\\eta$'),
           ('M3_em_1view', 'Single-view EM'),
           ('M4_em_2view', 'NIRA w/o name-hard'), ('M4n_em_namehard', '\\textbf{NIRA (ours)}')]


def get(model, split, cond, key, method, field):
    v = [r['results'][key]['agg'][method][field]['mean'] for (mo, sp, c, sd), r in runs.items()
         if mo == model and sp == split and c == cond and key in r['results']
         and method in r['results'][key]['agg'] and field in r['results'][key]['agg'][method]]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def size_fmt(x):
    return f'{x:,.0f}'.replace(',', '{,}') if x >= 100 else f'{x:.1f}'


def cov_size(model, split, cond, key, method, bold=False):
    c, n = get(model, split, cond, key, method, 'cov')
    s, _ = get(model, split, cond, key, method, 'size_mean')
    if c is None:
        return '--'
    txt = f'{c:.3f} / {size_fmt(s)}'
    return f'\\textbf{{{txt}}}' if bold else txt


def header(splits):
    top = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{lab}}}' for _, lab in splits)
    rules = ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(splits)))
    sub = ' & '.join('RULE & PMF' for _ in splits)
    return f'Method & {top} \\\\\n{rules}\n & {sub} \\\\\n\\midrule\n'


def method_rows(cond, key, methods, splits=SPLITS):
    lines = []
    for m, lab in methods:
        cells = [cov_size(model, sp, cond, key, m, bold=(m == 'M4n_em_namehard')) for sp, _ in splits for model in MODELS]
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines) + '\n'


def eta_rows(cond, key, splits=SPLITS, with_true=False, with_ablation=False):
    """noise-rate rows: realised rate (optional), estimate without the name-hard kernel (optional), NIRA's estimate"""
    rows = []
    if with_true:
        rows.append(('Realised noise rate $\\eta$', '_true_eta', 'mean', False))
    if with_ablation:
        rows.append(('$\\hat\\eta$, w/o name-hard kernel', 'M4_em_2view', 'eta_hat', False))
    rows.append(('$\\hat\\eta$, \\textbf{NIRA}', 'M4n_em_namehard', 'eta_hat', True))
    out = ''
    for lab, m, f, bold in rows:
        cells = []
        for sp, _ in splits:
            for model in MODELS:
                if m == '_true_eta':  # stored as agg['_true_eta']['mean']
                    t = [r['results'][key]['agg']['_true_eta']['mean'] for (mo, s2, c, sd), r in runs.items()
                         if mo == model and s2 == sp and c == cond]
                    v = float(np.mean(t)) if t else None
                else:
                    v, _ = get(model, sp, cond, key, m, f)
                txt = '--' if v is None else f'{v:.3f}'
                cells.append(f'\\textbf{{{txt}}}' if bold and v is not None else txt)
        out += f'{lab} & ' + ' & '.join(cells) + ' \\\\\n'
    return out


def write(name, body):
    # \bottomrule must sit in the same file as the last row: after \input returns, LaTeX no longer accepts \noalign
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write(body + '\\bottomrule\n')


# ---- Table: synthetic random noise (dot score), eta = 0.2 and 0.5 ----
body = header(SPLITS)
for cond, lab in [('syn0.2', '$\\eta=0.2$'), ('syn0.5', '$\\eta=0.5$')]:
    body += f'\\multicolumn{{7}}{{l}}{{\\emph{{Random seed noise, {lab}}}}} \\\\\n'
    body += method_rows(cond, 'dot|random|ncal3000', METHODS)
    body += eta_rows(cond, 'dot|random|ncal3000')
    body += '\\midrule\n' if cond == 'syn0.2' else ''
write('tab_synthetic.tex', body)

# ---- Table: structure-hard noise eta = 0.5 (dot score) ----
hard_methods = [m for m in METHODS if m[0] not in ('M2_rule_weight',)]
body = header(SPLITS) + method_rows('syn0.5', 'dot|hard|ncal3000', hard_methods)
write('tab_hard.tex', body)

# ---- Table: LLM annotation noise (dot score) ----
body = (header(SPLITS) + method_rows('llm', 'dot|llm|ncal3000', METHODS) + '\\midrule\n'
        + eta_rows('llm', 'dot|llm|ncal3000', with_true=True, with_ablation=True))
write('tab_llm.tex', body)

# ---- Table: decisions. Main cells use all test entities; the final row uses merged entities. ----
body = header(SPLITS)
for m, lab in [('M0_oracle', 'Clean-seed oracle$^{\\dagger}$'), ('M1_naive', 'Split CP'),
               ('M4_em_2view', 'NIRA w/o name-hard'), ('M4n_em_namehard', '\\textbf{NIRA (ours)}')]:
    cells = []
    for sp, _ in SPLITS:
        for model in MODELS:
            mg, _ = get(model, sp, 'llm', 'csls|llm|ncal3000', m, 'dec_merge')
            wr, _ = get(model, sp, 'llm', 'csls|llm|ncal3000', m, 'dec_wrong')
            txt = '--' if mg is None else f'{100 * mg:.1f} / {100 * wr:.1f}'
            cells.append(f'\\textbf{{{txt}}}' if m == 'M4n_em_namehard' else txt)
    body += f'{lab} & ' + ' & '.join(cells) + ' \\\\\n'
body += '\\midrule\n'
cells = []
for sp, _ in SPLITS:
    for model in MODELS:
        error, _ = get(model, sp, 'llm', 'csls|llm|ncal3000', 'M4n_em_namehard', 'dec_merge_err')
        cells.append(f'{100 * error:.1f}')
body += 'NIRA error among merged (\\%) & ' + ' & '.join(cells) + ' \\\\\n'
write('tab_decision.tex', body)

# ---- Table: CSLS score under synthetic noise (appendix) ----
body = header(SPLITS)
for cond, lab in [('syn0.2', '$\\eta=0.2$'), ('syn0.5', '$\\eta=0.5$')]:
    body += f'\\multicolumn{{7}}{{l}}{{\\emph{{Random seed noise, {lab}, CSLS score}}}} \\\\\n'
    body += method_rows(cond, 'csls|random|ncal3000', METHODS)
    body += '\\midrule\n' if cond == 'syn0.2' else ''
write('tab_csls_synthetic.tex', body)

# ---- Table: base-model accuracy (CSLS Hits@1) ----
body = header(SPLITS)
for cond, lab in [('syn0.2', 'Random noise, $\\eta=0.2$'), ('syn0.5', 'Random noise, $\\eta=0.5$'), ('llm', 'LLM-annotated seeds')]:
    cells = []
    for sp, _ in SPLITS:
        for model in MODELS:
            v = [r['hits1']['csls'] for (mo, s2, c, sd), r in runs.items() if mo == model and s2 == sp and c == cond]
            cells.append(f'{np.mean(v):.3f}' if v else '--')
    body += f'{lab} & ' + ' & '.join(cells) + ' \\\\\n'
write('tab_hits.tex', body)

# ---- FB15K-DB15K: columns = noise condition x base model, rows = methods ----
FB_COLS = [('syn0.2', 'dot|random|ncal3000', 'Random, $\\eta=0.2$'), ('syn0.5', 'dot|random|ncal3000', 'Random, $\\eta=0.5$'),
           ('syn0.5', 'dot|hard|ncal3000', 'Look-alike, $\\eta=0.5$'), ('llm', 'dot|llm|ncal3000', 'LLM annotation')]
if any(k[1] == 'fbdb' for k in runs):
    body = ''
    for bi, block in enumerate([FB_COLS[:2], FB_COLS[2:]]):  # two stacked blocks of four columns stay legible
        top = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{lab}}}' for _, _, lab in block)
        rules = ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(block)))
        body += ('\\midrule\n' if bi else '') + f'Method & {top} \\\\\n{rules}\n & ' + \
            ' & '.join('RULE & PMF' for _ in block) + ' \\\\\n\\midrule\n'
        for m, lab in METHODS:
            if m == 'M2_rule_weight':
                continue
            cells = [cov_size(model, 'fbdb', cond, key, m, bold=(m == 'M4n_em_namehard')) for cond, key, _ in block
                     for model in MODELS]
            body += f'{lab} & ' + ' & '.join(cells) + ' \\\\\n'
    write('tab_fbdb.tex', body)

# ---- robustness sweeps under LLM annotation noise (pilot/scripts/robust_sweep.py): coverage / set size relative to
# the clean-seed oracle, split CP and NIRA per benchmark; means over both base models and three training seeds
# (geometric mean for the ratio), 10 calibration draws each ----
rob = {}
for p in glob.glob(f'{CORRECTED}/robust/*_llm_s*.json'):
    m = re.match(r'(rule|pmf)_(zh_en|ja_en|fr_en|fbdb)_llm_s(\d+)\.json', os.path.basename(p))
    if m:
        rob.setdefault(m.group(2), []).append(json.load(open(p, encoding='utf-8')))
if rob:
    R0 = next(iter(rob.values()))[0]
    deltas, mn = R0['deltas'], R0['methods']

    def rob_cell(split, ncal, delta, method):
        covs, lr = [], []
        d_i, m_i, o_i = deltas.index(delta), mn.index(method), mn.index('M0_oracle')
        for R in rob.get(split, []):
            c = next(c for c in R['configs'] if c['ncal'] == ncal)
            covs.append(np.mean([s['cov'][d_i][m_i] for s in c['splits']]))
            lr.append(np.log(np.mean([s['size_mean'][d_i][m_i] for s in c['splits']])
                             / np.mean([s['size_mean'][d_i][o_i] for s in c['splits']])))
        if not covs:
            return '--'
        ratio = float(np.exp(np.mean(lr)))
        txt = f'{np.mean(covs):.3f} / ' + (f'{ratio:.2f}' if ratio < 10 else f'{ratio:.1f}' if ratio < 100 else size_fmt(ratio))
        return f'\\textbf{{{txt}}}' if method == 'M4n_em_namehard' else txt

    RSPLITS = SPLITS + [('fbdb', 'FB15K--DB15K')]
    top = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{lab}}}' for _, lab in RSPLITS)
    rules = ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(RSPLITS)))
    body = f'Setting & {top} \\\\\n{rules}\n & ' + ' & '.join('Split CP & NIRA' for _ in RSPLITS) + ' \\\\\n\\midrule\n'
    body += '\\multicolumn{9}{l}{\\emph{Target coverage $1-\\delta$, $n=3{,}000$ calibration seeds}} \\\\\n'
    for d in sorted(deltas, reverse=True):
        body += f'{1 - d:.2f} & ' + ' & '.join(rob_cell(sp, 3000, d, m) for sp, _ in RSPLITS
                                               for m in ('M1_naive', 'M4n_em_namehard')) + ' \\\\\n'
    body += '\\midrule\n\\multicolumn{9}{l}{\\emph{Calibration seeds $n$, target coverage 0.90}} \\\\\n'
    for nc in sorted({c['ncal'] for c in R0['configs']}):
        body += f'{nc:,}'.replace(',', '{,}') + ' & ' + ' & '.join(rob_cell(sp, nc, 0.10, m) for sp, _ in RSPLITS
                                                                   for m in ('M1_naive', 'M4n_em_namehard')) + ' \\\\\n'
    write('tab_robust_llm.tex', body)

# ---- sensitivity to NIRA's own hyperparameters (robust_sweep.py --mode hparam): one hyperparameter changed at a time,
# target 0.90, n = 3,000; means over the alignment models of each noise type (geometric mean for the set-size ratio) ----
hp = {'syn0.5': [], 'llm': []}
for p in glob.glob(f'{CORRECTED}/hparam/*.json'):
    m = re.match(r'(rule|pmf)_(zh_en|ja_en|fr_en|fbdb)_(syn0\.5|llm)_s(\d+)\.json', os.path.basename(p))
    if m:
        hp[m.group(3)].append(json.load(open(p, encoding='utf-8')))
B0, K0, PAIRS0, M0 = 40, 5, 20000, 5


def hp_stats(noise, bins=B0, k=K0, pairs=PAIRS0, hard_m=M0, method='M4n_em_namehard'):
    """(mean coverage, geometric-mean set size / oracle, mean eta_hat - eta, models) at target 0.90"""
    cov, lr, err = [], [], []
    for R in hp['llm' if noise == 'llm' else 'syn0.5']:
        c = next((c for c in R['configs'] if (c['noise'], c['bins'], c['k'], c['pairs'], c['hard_m'])
                  == (noise, bins, k, pairs, hard_m)), None)
        if c is None:
            continue
        d_i, m_i, o_i = R['deltas'].index(0.10), R['methods'].index(method), R['methods'].index('M0_oracle')
        cov.append(np.mean([s['cov'][d_i][m_i] for s in c['splits']]))
        lr.append(np.log(np.mean([s['size_mean'][d_i][m_i] for s in c['splits']])
                         / np.mean([s['size_mean'][d_i][o_i] for s in c['splits']])))
        err.append(np.mean([s['eta_hat'] - s['true_eta'] for s in c['splits']]))
    if not cov:
        return None
    return float(np.mean(cov)), float(np.exp(np.mean(lr))), float(np.mean(err)), len(cov)


def hp_cells(st, with_err=True, bold=False):
    if st is None:
        return ['--', '--'] if with_err else ['--']
    txt = f'{st[0]:.3f} / {st[1]:.2f}'
    out = [f'\\textbf{{{txt}}}' if bold else txt]
    return out + [f'${st[2]:+.3f}$'] if with_err else out


if hp['syn0.5'] or hp['llm']:
    NOISES = [('random', 'Random noise, $\\eta=0.5$'), ('hard', 'Look-alike noise, $\\eta=0.5$'), ('llm', 'LLM annotation noise')]
    top = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{lab}}}' for _, lab in NOISES)
    rules = ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(NOISES)))
    body = (f'Setting & {top} \\\\\n{rules}\n & ' + ' & '.join('Coverage / size & $\\hat\\eta-\\eta$' for _ in NOISES)
            + ' \\\\\n\\midrule\n')
    blocks = [('Quantile bins $B$', 'bins', [10, 20, 40, 80], B0, lambda v: f'{v}'),
              ('Neighbourhood size $k$ of the hard kernels', 'k', [2, 3, 5, 10, 20], K0, lambda v: f'{v}'),
              ('Random-kernel pairs', 'pairs', [2000, 5000, 20000, 50000], PAIRS0, lambda v: f'{v:,}'.replace(',', '{,}'))]
    for bi, (lab, field, vals, default, fmt) in enumerate(blocks):
        body += ('\\midrule\n' if bi else '') + f'\\multicolumn{{7}}{{l}}{{\\emph{{{lab}}}}} \\\\\n'
        for v in vals:
            cells = sum((hp_cells(hp_stats(nz, **{field: v})) for nz, _ in NOISES), [])
            body += fmt(v) + ('$^{*}$' if v == default else '') + ' & ' + ' & '.join(cells) + ' \\\\\n'
    write('tab_hparam.tex', body)
    # look-alike labels drawn from the m most similar wrong targets: baselines, NIRA's default kernel and a matched kernel
    body = ('Neighbourhood $m$ & Split CP & NACP, true $\\eta^{\\ddagger}$ & \\multicolumn{2}{c}{\\textbf{NIRA}, $k=5$} & '
            '\\multicolumn{2}{c}{NIRA, $k=m$} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
            ' & Coverage / size & Coverage / size & Coverage / size & $\\hat\\eta-\\eta$ & Coverage / size & $\\hat\\eta-\\eta$ \\\\\n'
            '\\midrule\n')
    for mv in [2, 5, 10, 20]:
        cells = (hp_cells(hp_stats('hard', hard_m=mv, method='M1_naive'), with_err=False)
                 + hp_cells(hp_stats('hard', hard_m=mv, method='B1_nacp_true_eta'), with_err=False)
                 + hp_cells(hp_stats('hard', hard_m=mv), bold=True)
                 + (['\\multicolumn{2}{c}{(same as $k=5$)}'] if mv == K0 else hp_cells(hp_stats('hard', k=mv, hard_m=mv))))
        body += f'{mv}' + ('$^{*}$' if mv == M0 else '') + ' & ' + ' & '.join(cells) + ' \\\\\n'
    write('tab_hard_m.tex', body)

# ---- summary numbers quoted in the text ----
summ = {}
for model in MODELS:
    for cond in ['syn0.2', 'syn0.5']:
        covs = [r['results']['dot|random|ncal3000']['agg']['M4n_em_namehard']['cov']['mean']
                for (mo, sp, c, sd), r in runs.items() if mo == model and c == cond and sp != 'fbdb']
        errs = [abs(r['results']['dot|random|ncal3000']['agg']['M4n_em_namehard']['eta_hat']['mean'] - float(cond[3:]))
                for (mo, sp, c, sd), r in runs.items() if mo == model and c == cond and sp != 'fbdb']
        summ[f'{model}_{cond}'] = dict(n=len(covs), cov_min=min(covs), cov_max=max(covs), eta_err_max=max(errs))
    covs = [r['results']['dot|llm|ncal3000']['agg']['M4n_em_namehard']['cov']['mean']
            for (mo, sp, c, sd), r in runs.items() if mo == model and c == 'llm' and sp != 'fbdb']
    summ[f'{model}_llm'] = dict(n=len(covs), cov_min=min(covs), cov_max=max(covs))


# robustness sweeps: per (sweep, noise, method) the range over alignment models and sweep values of the coverage minus
# the target and of the set size relative to the oracle; NIRA's noise-rate error along the noise-rate sweep
def rob_records(R):
    deltas, mn = R['deltas'], R['methods']
    o_i = mn.index('M0_oracle')
    for c in R['configs']:
        sweeps = ([('llm_level', d) for d in deltas] + [('llm_ncal', 0.10)] if c['noise'] == 'llm' else
                  ([('level', d) for d in deltas] if (c['eta'], c['ncal']) == (0.5, 3000) else [])
                  + ([('ncal', 0.10)] if c['eta'] == 0.5 else []) + ([('eta', 0.10)] if c['ncal'] == 3000 else []))
        for sweep, d in sweeps:
            if sweep == 'llm_level' and c['ncal'] != 3000:
                continue
            d_i = deltas.index(d)
            for m_i, m in enumerate(mn):
                yield dict(sweep=sweep, noise=c['noise'], eta=c['eta'], ncal=c['ncal'], delta=d, method=m,
                           gap=float(np.mean([s['cov'][d_i][m_i] for s in c['splits']]) - (1 - d)),
                           ratio=float(np.mean([s['size_mean'][d_i][m_i] for s in c['splits']])
                                       / np.mean([s['size_mean'][d_i][o_i] for s in c['splits']])),
                           eta_err=float(np.mean([s['eta_hat'] - s['true_eta'] for s in c['splits']])))


recs = []
for p in sorted(glob.glob(f'{CORRECTED}/robust/*.json')):
    for r in rob_records(json.load(open(p, encoding='utf-8'))):
        r['model'] = os.path.basename(p)[:-5]
        recs.append(r)
if recs:
    rs = {}
    for sweep in ('level', 'ncal', 'eta', 'llm_level', 'llm_ncal'):
        for noise in ('random', 'hard', 'llm'):
            for m in ('M1_naive', 'B1_nacp_true_eta', 'M4n_em_namehard'):
                sel = [r for r in recs if r['sweep'] == sweep and r['noise'] == noise and r['method'] == m]
                if sel:
                    rs[f'{sweep}|{noise}|{m}'] = dict(
                        n=len(sel), models=len({r['model'] for r in sel}),
                        gap=[round(min(r['gap'] for r in sel), 4), round(max(r['gap'] for r in sel), 4)],
                        ratio=[round(min(r['ratio'] for r in sel), 3), round(max(r['ratio'] for r in sel), 3)],
                        eta_err=[round(min(r['eta_err'] for r in sel), 4), round(max(r['eta_err'] for r in sel), 4)])
    for noise in ('random', 'hard'):  # NIRA at the smallest calibration set
        sel = [r for r in recs if r['sweep'] == 'ncal' and r['noise'] == noise and r['ncal'] == 250
               and r['method'] == 'M4n_em_namehard']
        if sel:
            rs[f'ncal250|{noise}|M4n_em_namehard'] = dict(gap=[round(min(r['gap'] for r in sel), 4), round(max(r['gap'] for r in sel), 4)],
                                                           ratio=[round(min(r['ratio'] for r in sel), 3), round(max(r['ratio'] for r in sel), 3)])
    summ['robust'] = rs
json.dump(summ, open(os.path.join(OUT, 'summary_numbers.json'), 'w'), indent=1)
print(json.dumps(summ, indent=1))
print('tables:', sorted(os.listdir(OUT)))
