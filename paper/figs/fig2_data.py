# Source data for Fig. 2 (mechanism): one calibration draw of 3,000 LLM-labelled seeds for RULE on ZH-EN (training
# seed 3408). Recomputes, with the estimator code of the analysis, the two scores of every seed, the label-free kernel
# samples and the EM posteriors with and without the name-hard kernel. Writes source_data/fig2_*.csv.
import json, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp

PILOT = str(Path(__file__).resolve().parents[2] / 'pilot')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source_data')
os.makedirs(OUT, exist_ok=True)
SPLIT, EXPORT = 'zh_en', f'{PILOT}/runs/ext/rule_zh_en_llm_s3408.npz'
rng = np.random.default_rng(7)  # one illustrative calibration draw

E = np.load(EXPORT)
ref = E['test_ill'].astype(np.int64)
pos = {int(e): i for i, e in enumerate(E['ent_idx'])}
J = E['joint'].astype(np.float32)
L, R = J[[pos[int(e)] for e in ref[:, 0]]], J[[pos[int(e)] for e in ref[:, 1]]]
n = len(ref)

# name view: character-bigram TF cosine of the translated names (as in the analysis)
names = dict((int(i), t) for i, t in json.load(open(f'{PILOT}/data/DBP15K/translated_ent_name/dbp_{SPLIT}.json', encoding='utf-8')))
vocab, rows, cols, vals = {}, [], [], []
for r_, e in enumerate(np.concatenate([ref[:, 0], ref[:, 1]])):
    cnt = {}
    for w in names.get(int(e), []):
        w = w.lower()
        for j in range(len(w) - 1):
            cnt[w[j:j + 2]] = cnt.get(w[j:j + 2], 0) + 1
    for g, c in cnt.items():
        rows.append(r_); cols.append(vocab.setdefault(g, len(vocab))); vals.append(c)
NV = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(2 * n, len(vocab)))
nrm = np.sqrt(np.asarray(NV.multiply(NV).sum(1)).ravel()); nrm[nrm == 0] = 1
NV = sp.diags(1 / nrm) @ NV
NL, NR = NV[:n].tocsr(), NV[n:].tocsr()
vsim = lambda i, j: np.asarray(NL[i].multiply(NR[j]).sum(1)).ravel().astype(np.float32)
ssim = lambda i, j: np.einsum('ij,ij->i', L[i], R[j])

# LLM calibration labels and one draw of 3,000 seeds
tgt_pos = {int(t): j for j, t in enumerate(ref[:, 1])}
llm = {}
for l in open(f'{PILOT}/runs/ext/llm/annot_{SPLIT}.jsonl', encoding='utf-8'):
    r_ = json.loads(l)
    if r_['part'] == 'cal' and r_['choice'] is not None:
        llm[int(r_['ref_index'])] = tgt_pos[int(r_['choice'])]
pool = np.array(sorted(llm), np.int64)
cal = np.sort(rng.choice(pool, 3000, replace=False))
obs = np.array([llm[int(i)] for i in cal], np.int64)
wrong = obs != cal
s_obs, v_obs = ssim(cal, obs), vsim(cal, obs)

# label-free kernels: random pairs, 5 structure-hard and 5 name-hard candidates per seed source (observed target excluded)
ri = rng.choice(cal, 20000); rj = rng.integers(0, n, 20000)
s_r, v_r = ssim(ri, rj), vsim(ri, rj)
NRt = NR.T.tocsc()
hi, hj, vi, vj = [], [], [], []
for s0 in range(0, len(cal), 500):
    blk = cal[s0:s0 + 500]
    S_blk = L[blk] @ R.T
    V_blk = (NL[blk] @ NRt).toarray()
    for k, i in enumerate(blk):
        o = obs[s0 + k]
        top = [t for t in np.argpartition(-S_blk[k], 6)[:6] if t != o][:5]
        hi += [i] * len(top); hj += top
        topv = [t for t in np.argpartition(-V_blk[k], 6)[:6] if t != o][:5]
        vi += [i] * len(topv); vj += topv
hi, hj, vi, vj = map(np.array, (hi, hj, vi, vj))
s_h, v_h = ssim(hi, hj), vsim(hi, hj)
s_vh, v_vh = ssim(vi, vj), vsim(vi, vj)


# ---- estimator code copied from the analysis (edges, hist, look, em, em_multi) ----
def edges(x, nb=40):
    e = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    if len(e) < 3: e = np.linspace(x.min() - 1e-6, x.max() + 1e-6, nb + 1)
    e[0], e[-1] = -np.inf, np.inf
    return e


def hist(x, e, w=None):
    h, _ = np.histogram(x, bins=e, weights=w); h = h + 1e-3
    return h / h.sum()


def look(h, x, e):
    return h[np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(h) - 1)]


def em_multi(s, v, kernels, iters=300):
    es = edges(np.concatenate([s] + [k[0] for k in kernels])); ev = edges(np.concatenate([v] + [k[1] for k in kernels]))
    F = [(hist(ks, es), hist(kv, ev)) for ks, kv in kernels]
    a = np.stack([look(f, s, es) * look(g, v, ev) for f, g in F])
    p = np.full(len(s), 0.8); beta = np.full(len(kernels), 1.0 / len(kernels)); e = 0.2
    for _ in range(iters):
        fc = hist(s, es, p); gc = hist(v, ev, p)
        mix = (beta[:, None] * a).sum(0)
        lc = (1 - e) * look(fc, s, es) * look(gc, v, ev)
        p = lc / (lc + e * mix)
        resp = (1 - p)[None, :] * beta[:, None] * a / mix[None, :]
        e = float(np.clip(1 - p.mean(), 1e-3, 0.95))
        beta = np.clip(resp.sum(1) / max((1 - p).sum(), 1e-9), 0.01, 0.99); beta = beta / beta.sum()
    return p, e, beta


p_wo, e_wo, b_wo = em_multi(s_obs, v_obs, [(s_r, v_r), (s_h, v_h)])
p_nira, e_nira, b_nira = em_multi(s_obs, v_obs, [(s_r, v_r), (s_h, v_h), (s_vh, v_vh)])

s_true = ssim(cal, cal)  # score of the correct counterpart (what a clean-seed oracle calibrates on)
np.savetxt(f'{OUT}/fig2_seeds.csv', np.column_stack([s_obs, v_obs, wrong.astype(int), p_wo, p_nira, s_true]), delimiter=',',
           header='s,v,wrong,posterior_wo_namehard,posterior_nira,s_true_pair', comments='',
           fmt=['%.6f', '%.6f', '%d', '%.6f', '%.6f', '%.6f'])
ker = np.concatenate([np.column_stack([s_r, v_r, np.zeros_like(s_r)]), np.column_stack([s_h, v_h, np.ones_like(s_h)]),
                      np.column_stack([s_vh, v_vh, 2 * np.ones_like(s_vh)])])
np.savetxt(f'{OUT}/fig2_kernels.csv', ker, delimiter=',', header='s,v,kernel(0=random,1=structure-hard,2=name-hard)',
           comments='', fmt=['%.6f', '%.6f', '%d'])
json.dump(dict(export=os.path.basename(EXPORT), n_seeds=int(len(cal)), wrong_rate=float(wrong.mean()),
               eta_hat_wo=e_wo, eta_hat_nira=e_nira, beta_wo=b_wo.tolist(), beta_nira=b_nira.tolist(),
               kernel_sizes=[int(len(s_r)), int(len(s_h)), int(len(s_vh))]),
          open(f'{OUT}/fig2_summary.json', 'w'), indent=1)
print(f'wrong {wrong.mean():.3f} | eta_hat w/o {e_wo:.3f} NIRA {e_nira:.3f} | beta NIRA {np.round(b_nira, 3)}')
for lab, p in [('w/o name-hard', p_wo), ('NIRA', p_nira)]:
    print(f'{lab}: mean posterior correct {p[~wrong].mean():.3f} wrong {p[wrong].mean():.3f} | '
          f'share of weight on wrong seeds {p[wrong].sum() / p.sum():.3f}')
