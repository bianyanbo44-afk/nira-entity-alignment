"""Paired audit of original and corrected calibration outputs, without retraining."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/label_free_v2'
ap = argparse.ArgumentParser()
ap.add_argument('--partial', action='store_true')
args = ap.parse_args()
rows = []
fixed = ('M0_oracle', 'M1_naive', 'M2_rule_weight', 'M2b_rule_filter')
fixed_max = 0.
fixed_by_field = {}
splits_checked = 0
for model in ('rule', 'pmf'):
    for p in sorted((OUT / 'main' / model).glob('*_results.json')):
        split, condition, seed = p.name.replace('_results.json', '').rsplit('_', 2)
        old_dir = ('fbdb/res_rule' if model == 'rule' else 'fbdb/res_pmf') if split == 'fbdb' else (
            'ext/res_b' if model == 'rule' else 'ext/res_pmf_b')
        old_path = ROOT / 'runs' / old_dir / p.name
        new, old = json.loads(p.read_text(encoding='utf-8')), json.loads(old_path.read_text(encoding='utf-8'))
        for key, result in new['results'].items():
            assert len(result['splits']) == 20, (p, key)
            original = old['results'][key]
            for a, b in zip(result['splits'], original['splits']):
                splits_checked += 1
                for method in fixed:
                    for field, value in a[method].items():
                        difference = abs(value - b[method][field])
                        if np.isfinite(difference):
                            fixed_max = max(fixed_max, difference)
                            fixed_by_field[field] = max(fixed_by_field.get(field, 0.), difference)
                        # The paired archived outputs can have float32 score-boundary drift.
                        # Strict same-input equivalence is checked by --verify-all separately.
                        if field == 'cov':
                            np.testing.assert_allclose(value, b[method][field], rtol=0, atol=2 / 7000,
                                                       err_msg=f'{p.name}:{key}:{method}:{field}')
                        if field == 'thr' and np.isfinite(difference):
                            assert difference < 5e-6, (p, key, method, difference)
                assert a['_true_eta'] == b['_true_eta'], (p, key)
            a, b = result['agg']['M4n_em_namehard'], original['agg']['M4n_em_namehard']
            rows.append(dict(model=model, benchmark=split, training_noise=condition, training_seed=seed,
                             setting=key, old_coverage=b['cov']['mean'], coverage=a['cov']['mean'],
                             coverage_change=a['cov']['mean'] - b['cov']['mean'],
                             old_set_size=b['size_mean']['mean'], set_size=a['size_mean']['mean'],
                             old_eta=b['eta_hat']['mean'], eta=a['eta_hat']['mean'],
                             merge=a['dec_merge']['mean'], wrong_all=a['dec_wrong']['mean'],
                             wrong_among_merged=a['dec_merge_err']['mean'], loss=a['dec_loss']['mean']))

main_files = list((OUT / 'main').glob('*/*_results.json'))
sweep_counts = {mode: len(list((OUT / mode).glob('*.json'))) for mode in ('robust', 'hparam')}
if not args.partial:
    assert len(main_files) == 72, len(main_files)
    assert sweep_counts == dict(robust=48, hparam=48), sweep_counts
main = [r for r in rows if r['setting'].startswith('dot|') and not (
    '|hard|' in r['setting'] and r['training_noise'] == 'syn0.2')]
decisions = [r for r in rows if r['benchmark'] != 'fbdb' and r['setting'] == 'csls|llm|ncal3000']
summary = dict(trained_models=len(main_files), main_evaluation_settings=len(main),
               repeated_splits_checked=splits_checked, sweep_counts=sweep_counts,
               unchanged_baseline_max_absolute_difference=fixed_max,
               unchanged_baseline_max_difference_by_field=fixed_by_field,
               coverage_range=[min(r['coverage'] for r in main), max(r['coverage'] for r in main)],
               max_absolute_coverage_change=max(abs(r['coverage_change']) for r in rows),
               mean_absolute_coverage_change=float(np.mean([abs(r['coverage_change']) for r in rows])))
if decisions:
    summary['llm_dbp_decisions'] = {name: [min(r[name] for r in decisions), max(r[name] for r in decisions)]
                                     for name in ('merge', 'wrong_all', 'wrong_among_merged')}
    grouped = []
    for model in ('rule', 'pmf'):
        for benchmark in ('zh_en', 'ja_en', 'fr_en'):
            group = [r for r in decisions if r['model'] == model and r['benchmark'] == benchmark]
            if len(group) == 3:
                grouped.append(dict(model=model, benchmark=benchmark,
                                    **{field: float(np.mean([r[field] for r in group]))
                                       for field in ('merge', 'wrong_all', 'wrong_among_merged', 'loss')}))
    summary['headline_decision_averages'] = grouped
for mode in ('robust', 'hparam'):
    changes = []; mean_changes = []; count = 0; unaffected = 0.; baseline_by_metric = {}
    for p in sorted((OUT / mode).glob('*.json')):
        a = json.loads(p.read_text(encoding='utf-8')); b = json.loads((ROOT / 'runs' / mode / p.name).read_text(encoding='utf-8'))
        assert a['methods'] == b['methods'] and a['deltas'] == b['deltas']
        assert len(a['configs']) == len(b['configs'])
        ni = a['methods'].index('M4n_em_namehard')
        indices = [a['methods'].index(m) for m in fixed]
        for ca, cb in zip(a['configs'], b['configs']):
            for field in ('noise', 'eta', 'ncal', 'bins', 'k', 'pairs', 'hard_m'):
                default = dict(bins=40, k=5, pairs=20000, hard_m=5).get(field)
                assert ca[field] == cb.get(field, default), (p, field)
            mean_a = np.mean([np.array(s['cov'])[:, ni] for s in ca['splits']], axis=0)
            mean_b = np.mean([np.array(s['cov'])[:, ni] for s in cb['splits']], axis=0)
            mean_changes.extend((mean_a - mean_b).tolist())
            for sa, sb in zip(ca['splits'], cb['splits']):
                count += 1
                x, y = np.array(sa['cov']), np.array(sb['cov'])
                changes.extend((x[:, ni] - y[:, ni]).tolist())
                for metric in ('cov', 'size_mean', 'size_median', 'singleton', 'empty'):
                    x, y = np.array(sa[metric])[:, indices], np.array(sb[metric])[:, indices]
                    if metric in ('cov', 'singleton', 'empty'):
                        np.testing.assert_allclose(x, y, atol=2 / (a['n'] - ca['ncal']), rtol=0)
                    unaffected = max(unaffected, float(np.abs(x-y).max()))
                    baseline_by_metric[metric] = max(baseline_by_metric.get(metric, 0.), float(np.abs(x-y).max()))
    summary[mode] = dict(calibration_config_draws=count,
                         max_absolute_nira_coverage_change=max(map(abs, changes)) if changes else None,
                         max_absolute_mean_coverage_change=max(map(abs, mean_changes)) if mean_changes else None,
                         unchanged_baseline_max_absolute_difference=unaffected,
                         unmodified_baseline_difference_by_metric=baseline_by_metric)
name = 'partial_audit' if args.partial else 'paired_audit'
(OUT / f'{name}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
with (OUT / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps(summary, indent=2))
