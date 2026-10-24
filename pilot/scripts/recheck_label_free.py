"""Recompute all paper results after removing gold-based kernel filtering.

Original outputs are preserved. Corrected main results and both complete sweeps
are written to runs/label_free_v2. Each child has a separate log; failures stop
the driver, and completed outputs can be resumed without rerunning training.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'label_free_v2'
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--stage', choices=('all', 'main', 'robust', 'hparam'), default='all')
args = ap.parse_args()
jobs = []
for model in ('rule', 'pmf'):
    for split in ('zh_en', 'ja_en', 'fr_en', 'fbdb'):
        for condition in ('syn0.2', 'syn0.5', 'llm'):
            for seed in (3408, 1234, 2025):
                fb = split == 'fbdb'
                stem = f'{model}_{split}_{condition}_s{seed}'
                export = ROOT / 'runs' / ('fbdb' if fb else 'ext') / f'{stem}.npz'
                labels = ROOT / 'runs' / ('fbdb/llm/annot_fbdb.jsonl' if fb else f'ext/llm/annot_{split}.jsonl')
                flags = ['--split', 'norm' if fb else split]
                if fb: flags += ['--names', str(ROOT / 'data/FBDB15K/entity_names/tokens_norm.json')]
                if condition == 'llm': flags += ['--cal_labels', str(labels)]
                if args.stage in ('all', 'main'):
                    prefix = OUT / 'main' / model / f'{split}_{condition}_s{seed}'
                    command = [str(ROOT / 'scripts/analyze_ext.py'), str(export), str(ROOT / 'data'),
                               str(prefix), '0.5' if condition == 'llm' else condition[3:]] + flags
                    jobs.append(('main/' + stem, command, Path(str(prefix) + '_results.json')))
                if condition != 'syn0.2':
                    for mode in ('robust', 'hparam'):
                        if args.stage not in ('all', mode): continue
                        result = OUT / mode / f'{stem}.json'
                        command = [str(ROOT / 'scripts/robust_sweep.py'), str(export), str(ROOT / 'data'),
                                   str(result), '--mode', mode] + flags
                        jobs.append((mode + '/' + stem, command, result))

OUT.mkdir(parents=True, exist_ok=True)
logs = OUT / 'logs'; logs.mkdir(exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
start = time.time()


def run(job):
    name, command, result = job
    if result.exists():
        with result.open(encoding='utf-8') as handle: json.load(handle)
        return name, 'cached'
    result.parent.mkdir(parents=True, exist_ok=True)
    log = logs / (name.replace('/', '_') + '.log')
    with log.open('w', encoding='utf-8') as handle:
        completed = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/run_calibration_fast.py')] + command,
                                   stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT.parent, env=env)
    if completed.returncode:
        raise RuntimeError(f'{name} failed ({completed.returncode}); see {log}')
    with result.open(encoding='utf-8') as handle: data = json.load(handle)
    data['reanalysis'] = dict(random_kernel='all uniform candidate draws; no gold-pair filtering',
                              training='existing frozen export', runtime='cached bins and exact sorted-row counts')
    result.write_text(json.dumps(data, indent=1), encoding='utf-8')
    return name, 'done'


print(f'{len(jobs)} analyses, {args.workers} workers; original results preserved', flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = [pool.submit(run, job) for job in jobs]
    for number, future in enumerate(as_completed(futures), 1):
        name, state = future.result()
        print(f'[{(time.time()-start)/60:.1f} min] {number}/{len(jobs)} {name}: {state}', flush=True)
print('All requested analyses completed.', flush=True)
