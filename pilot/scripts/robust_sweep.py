# Robustness sweeps for the paper's robustness figure: target coverage level, calibration-set size and calibration-noise
# rate, on one existing export and the inner-product score. The estimators are analyze_ext.py's, copied verbatim, and
# every configuration replays analyze_ext's random-draw order from a fresh generator, so the configuration
# (random noise, eta = the export's eta, ncal 3000) at delta 0.10 reproduces the first --splits splits of analyze_ext's
# dot|random rows up to floating-point rounding in the matrix products. Set sizes and NACP's F_r are counted on
# row-sorted scores instead of one n x n pass per threshold (same counts; --check compares both ways on the first split).
# --mode hparam instead varies NIRA's own hyperparameters one at a time at eta 0.5 (or the LLM labels) and ncal 3000:
# histogram bins B, the neighbourhood size k of the structure-hard and name-hard kernels, and the number of random-kernel
# pairs; for look-alike noise also the neighbourhood m the wrong labels are drawn from (top-m under s instead of top-5),
# with the default kernel and with k = m. Calibration draws and noisy labels come before every kernel draw, so all
# hyperparameter values see the same draws, and the default (40, 5, 20000, m 5) reproduces the robustness default exactly.
# usage: python -I robust_sweep.py <export.npz> <data_dir> <out.json> [--split zh_en] [--names tokens.json]
#        [--cal_labels annot.jsonl] [--splits 10] [--mode robust|hparam] [--only kind:eta:ncal[:B:k:pairs:m],...] [--check]
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('export'); ap.add_argument('data_dir'); ap.add_argument('out')
ap.add_argument('--splits', type=int, default=10); ap.add_argument('--seed', type=int, default=20261022)
ap.add_argument('--chunk', type=int, default=1000)
ap.add_argument('--split', default='zh_en'); ap.add_argument('--cal_labels', default='')
ap.add_argument('--names', default='', help='name-token file for the second view (default: DBP15K translated names)')
ap.add_argument('--only', default='', help='comma-separated kind:eta:ncal[:B:k:pairs] configurations (default: the full sweep)')
ap.add_argument('--mode', default='robust', choices=('robust', 'hparam'))
ap.add_argument('--check', action='store_true')
args = ap.parse_args()
DELTAS = (0.25, 0.20, 0.15, 0.10, 0.05, 0.02)  # target coverage 0.75 ... 0.98
NCALS = (250, 500, 1000, 2000, 3000)
ETAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
ETA0, NCAL0 = 0.5, 3000
BINS, KS, PAIRS, HARD_MS = (10, 20, 40, 80), (2, 3, 5, 10, 20), (2000, 5000, 20000, 50000), (2, 5, 10, 20)
B0, K0, PAIRS0, M0 = 40, 5, 20000, 5
KSET = KS if args.mode == 'hparam' else (K0,)
MSET = HARD_MS if args.mode == 'hparam' else (M0,)
METHODS = ('M0_oracle', 'M1_naive', 'M2_rule_weight', 'M2b_rule_filter', 'M3_em_1view', 'M4n_em_namehard',
           'B1_nacp_true_eta', 'B2_nacp_mpe_eta')
t0 = time.time()
E = np.load(args.export)
ref = E['test_ill'].astype(np.int64)
ent_idx = E['ent_idx'].astype(np.int64)
pos = {int(e): i for i, e in enumerate(ent_idx)}
J = E['joint'].astype(np.float32)
L = J[[pos[int(e)] for e in ref[:, 0]]]
R = J[[pos[int(e)] for e in ref[:, 1]]]
n = len(ref)
del J

# ---- inner-product scores in memory, their row-sorted copy, and the near-neighbour lists analyze_ext reads per row ----
S = np.empty((n, n), np.float32)
for s in range(0, n, args.chunk):
    S[s:s + args.chunk] = L[s:s + args.chunk] @ R.T
hits1 = float(np.mean(S.argmax(1) == np.arange(n)))
Ssort = np.sort(S, axis=1)
top_s = {k: np.empty((n, k + 1), np.int64) for k in KSET}  # structure-hard kernel: np.argpartition(-row, k+1)[:k+1]
hard = {m: np.empty((n, m), np.int64) for m in MSET}  # look-alike noise: np.argpartition(-row, m)[:m] with row[i] = -inf
for s in range(0, n, args.chunk):
    X = -S[s:s + args.chunk]
    for k in KSET:
        top_s[k][s:s + args.chunk] = np.argpartition(X, k + 1, axis=1)[:, :k + 1]
    X[np.arange(len(X)), np.arange(s, s + len(X))] = np.inf
    for m in MSET:
        hard[m][s:s + args.chunk] = np.argpartition(X, m, axis=1)[:, :m]
print(f'[{time.time()-t0:.0f}s] scores ready, n={n}, Hits@1 {hits1:.4f}', flush=True)

# ---- second view: character-bigram name vectors ----
names = dict((int(i), toks) for i, toks in json.load(open(args.names or os.path.join(
    args.data_dir, 'DBP15K', 'translated_ent_name', f'dbp_{args.split}.json'), encoding='utf-8')))
vocab, rows_, cols_, vals_ = {}, [], [], []
ents = np.concatenate([ref[:, 0], ref[:, 1]])
for r, e in enumerate(ents):
    cnt = {}
    for w in names.get(int(e), []):
        w = w.lower()
        for j in range(len(w) - 1):
            cnt[w[j:j + 2]] = cnt.get(w[j:j + 2], 0) + 1
    for g, c in cnt.items():
        rows_.append(r); cols_.append(vocab.setdefault(g, len(vocab))); vals_.append(c)
NV = sp.csr_matrix((np.array(vals_, np.float32), (rows_, cols_)), shape=(len(ents), max(len(vocab), 1)))
norm = np.sqrt(np.asarray(NV.multiply(NV).sum(1)).ravel()); norm[norm == 0] = 1
NV = sp.diags(1 / norm) @ NV
NL, NR = NV[:n], NV[n:]


def vsim(i, j):
    return np.asarray(NL[i].multiply(NR[j]).sum(1)).ravel().astype(np.float32)


Ln = L / np.linalg.norm(L, axis=1, keepdims=True)
Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
top_v = {k: np.empty((n, k + 1), np.int64) for k in KSET}  # name-hard kernel
NRt = NR.T.tocsc()
for s in range(0, n, args.chunk):
    X = (NL[s:s + args.chunk] @ NRt).toarray()
    for k in KSET:
        top_v[k][s:s + args.chunk] = np.argpartition(-X, k + 1, axis=1)[:, :k + 1]
print(f'[{time.time()-t0:.0f}s] name view ready', flush=True)


# ---- estimators, verbatim from analyze_ext.py ----
def rule_weight(i, j, rng, bs=512, tau=0.07, k=49):
    """RULE's get_weight reliability on the (normalised) joint embedding, averaged over both directions."""
    W = np.empty(len(i), np.float32); order = rng.permutation(len(i))
    for s in range(0, len(i), bs):
        b = order[s:s + bs]
        cs = Ln[i[b]] @ Rn[j[b]].T
        posv = np.diag(cs).copy()
        out = []
        for M in (cs, cs.T):
            neg = M.copy(); np.fill_diagonal(neg, -8.0)
            kk = min(k, neg.shape[1] - 1)
            raw = np.concatenate([posv[:, None], -np.sort(-neg, axis=1)[:, :kk]], axis=1)
            ev = np.exp(np.tanh(raw) / tau)
            u = raw.shape[1] / (ev + 1).sum(1)
            out.append((1 - u) / 2 + np.clip(posv, 0, 1) / 2)
        W[b] = (out[0] + out[1]) / 2
    return W


def wquantile(a, w, level):
    if not np.any(w > 0):
        return np.inf
    """smallest a* with sum_{a_i<=a*} w_i / (sum w + max w) >= level ; +inf if never."""
    o = np.argsort(a); a, w = a[o], w[o]
    c = np.cumsum(w) / (w.sum() + w.max())
    k = np.searchsorted(c, level)
    return a[k] if k < len(a) else np.inf


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


def em(s, v, sr, vr, sh, vh, use_view, iters=300):
    es = edges(np.concatenate([s, sr, sh])); ev = edges(np.concatenate([v, vr, vh]))
    fr, fh = hist(sr, es), hist(sh, es); gr, gh = hist(vr, ev), hist(vh, ev)
    p = np.full(len(s), 0.8); beta, e = 0.5, 0.2
    for _ in range(iters):
        fc = hist(s, es, p); gc = hist(v, ev, p)
        a_r = look(fr, s, es) * (look(gr, v, ev) if use_view else 1)
        a_h = look(fh, s, es) * (look(gh, v, ev) if use_view else 1)
        lc = (1 - e) * look(fc, s, es) * (look(gc, v, ev) if use_view else 1)
        ln = e * (beta * a_r + (1 - beta) * a_h)
        p = lc / (lc + ln)
        resp_r = (1 - p) * beta * a_r / (beta * a_r + (1 - beta) * a_h)
        e = float(np.clip(1 - p.mean(), 1e-3, 0.95))
        beta = float(np.clip(resp_r.sum() / max((1 - p).sum(), 1e-9), 0.01, 0.99))
    return p, e, beta


def em_multi(s, v, kernels, iters=300, nb=40):
    """M4n: two-view EM with several noise components (list of (s_samples, v_samples))."""
    es = edges(np.concatenate([s] + [k[0] for k in kernels]), nb)
    ev = edges(np.concatenate([v] + [k[1] for k in kernels]), nb)
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


def mpe_deconv(a_obs, a_rand, level, eta_fixed=None, kappa=0.2):
    """nonconformity a = -s; noise lives in the upper tail of a. Returns (threshold, eta_hat)."""
    grid = np.unique(np.concatenate([a_obs, np.quantile(a_rand, np.linspace(0, 1, 2001))]))
    so = 1 - np.searchsorted(np.sort(a_obs), grid, side='right') / len(a_obs)
    sr = 1 - np.searchsorted(np.sort(a_rand), grid, side='right') / len(a_rand)
    if eta_fixed is None:
        ok = sr >= kappa
        eta_hat = float(np.clip(np.min(so[ok] / sr[ok]), 0, 0.95))
    else:
        eta_hat = eta_fixed
    Fo, Fr = 1 - so, 1 - sr
    Fc = np.maximum.accumulate(np.clip((Fo - eta_hat * Fr) / max(1 - eta_hat, 1e-6), 0, 1))
    k = np.searchsorted(Fc, level)
    return (grid[k] if k < len(grid) else np.inf), eta_hat


def nacp_threshold(a_sorted, Fr_all, k, eps, delta):
    """NACP (Penso, Goldberger and Fetaya, COPA 2025, Algorithm 1); see analyze_ext.py."""
    n_ = len(a_sorted)
    q1 = np.quantile(a_sorted, float(np.clip((1 - delta) * (1 - eps) / (1 - eps / k), 0, 1)))
    q2 = np.quantile(a_sorted, float(np.clip((1 - delta) + delta * eps, 0, 1)))
    sel = (a_sorted >= q1) & (a_sorted <= q2)
    grid, Fr = a_sorted[sel], Fr_all[sel]
    if len(grid) == 0:
        return q2
    Fn = np.searchsorted(a_sorted, grid, side='right') / n_
    ok = (Fn - eps * Fr) / max(1 - eps, 1e-6) >= 1 - delta
    if ok.all():
        return grid[0]
    if not ok.any():
        return grid[-1]
    up = np.where(ok[1:] & ~ok[:-1])[0] + 1
    return grid[up[-1]] if len(up) else grid[int(np.argmax(ok))]


# ---- counting on the row-sorted scores ----
def random_label_cdf(cal, grid):
    """analyze_ext's F_r on the sorted grid: share of all (calibration source, candidate) pairs with -S <= grid."""
    c = np.zeros(len(grid), np.int64); neg = -grid.astype(np.float64)
    for i in cal:
        c += n - np.searchsorted(Ssort[i], neg, side='left')
    return c / (len(cal) * n)


def set_sizes(test, thr):
    """|C(e)| = #{t : S[e,t] >= thr} for every test row and every threshold in thr."""
    out = np.empty((len(test), len(thr)), np.int32)
    for r, e in enumerate(test):
        out[r] = n - np.searchsorted(Ssort[e], thr, side='left')
    return out


# ---- LLM calibration labels (optional) ----
llm_obs = None
if args.cal_labels:
    tgt_pos = {int(t): j for j, t in enumerate(ref[:, 1])}
    llm_obs = {}
    for l in open(args.cal_labels, encoding='utf-8'):
        r_ = json.loads(l)
        if r_['part'] == 'cal' and r_['choice'] is not None:
            llm_obs[int(r_['ref_index'])] = tgt_pos[int(r_['choice'])]
    pool_idx = np.array(sorted(llm_obs), np.int64)
    print(f'LLM calibration pool: {len(pool_idx)} annotated pairs, realised noise '
          f'{np.mean([llm_obs[i] != i for i in pool_idx]):.3f}', flush=True)

if args.only:
    configs = []
    for x in args.only.split(','):
        p = x.split(':') + [B0, K0, PAIRS0, M0][len(x.split(':')) - 3:]
        configs.append((p[0], None if p[1] == 'none' else float(p[1]), *map(int, p[2:7])))
elif args.mode == 'hparam':
    configs = []
    for kind, eta in ([('llm', None)] if args.cal_labels else [('random', ETA0), ('hard', ETA0)]):
        configs += ([(kind, eta, NCAL0, b, K0, PAIRS0, M0) for b in BINS] +
                    [(kind, eta, NCAL0, B0, k, PAIRS0, M0) for k in KS if k != K0] +
                    [(kind, eta, NCAL0, B0, K0, pr, M0) for pr in PAIRS if pr != PAIRS0])
    if not args.cal_labels:  # look-alike noise from other neighbourhoods, default kernel and matched kernel
        configs += [('hard', ETA0, NCAL0, B0, k, PAIRS0, m) for m in HARD_MS if m != M0 for k in (K0, m)]
elif args.cal_labels:
    configs = [('llm', None, nc, B0, K0, PAIRS0, M0) for nc in NCALS]
else:
    configs = ([(kind, eta, NCAL0, B0, K0, PAIRS0, M0) for kind in ('random', 'hard') for eta in ETAS] +
               [(kind, ETA0, nc, B0, K0, PAIRS0, M0) for kind in ('random', 'hard') for nc in NCALS if nc != NCAL0])
assert all(c[4] in KSET and c[6] in MSET for c in configs), 'neighbourhood size without precomputed neighbour lists'

results = []
for kind, eta, ncal, nbins, kk, npairs, mm in configs:
    rng = np.random.default_rng(args.seed)
    runs = []
    for sp_i in range(args.splits):
        if kind == 'llm':
            perm = rng.permutation(pool_idx); cal = perm[:ncal]
            test = np.setdiff1d(np.arange(n), cal)
            obs = np.array([llm_obs[int(i)] for i in cal], np.int64)
        else:
            perm = rng.permutation(n); cal, test = perm[:ncal], perm[ncal:]
            obs = cal.copy()
            k_noisy = int(len(cal) * eta)
            sel = rng.choice(len(cal), k_noisy, replace=False)
            if kind == 'random':
                obs[sel] = obs[sel][rng.permutation(k_noisy)]
            else:
                for kq in sel:
                    obs[kq] = rng.choice(hard[mm][cal[kq]])
        noisy_true = obs != cal
        s_obs = S[cal, obs]
        s_clean = S[cal, cal]
        v_obs = vsim(cal, obs)
        # All sampled pairs are retained; gold labels are used only for corruption and evaluation.
        ri = rng.choice(cal, npairs); rj = rng.integers(0, n, npairs)
        s_r = S[ri, rj]; v_r = vsim(ri, rj)
        hi, hj = [], []
        for kq, i in enumerate(cal):
            top = [t for t in top_s[kk][i] if t != obs[kq]][:kk]
            hi += [i] * len(top); hj += top
        hi, hj = np.array(hi), np.array(hj)
        s_h = S[hi, hj]; v_h = vsim(hi, hj)
        W = rule_weight(cal, obs, rng)
        keep = W >= 0.5
        a_keep = -s_obs[keep]
        p1, e1, b1 = em(s_obs, v_obs, s_r, v_r, s_h, v_h, use_view=False)
        vi = np.repeat(cal, kk)
        vj = np.array([[t for t in top_v[kk][i] if t != obs[kq]][:kk] for kq, i in enumerate(cal)]).ravel()
        s_vh = S[vi, vj]; v_vh = vsim(vi, vj)
        p4n, e4n, b4n = em_multi(s_obs, v_obs, [(s_r, v_r), (s_h, v_h), (s_vh, v_vh)], nb=nbins)
        _, e5 = mpe_deconv(-s_obs, -s_r, (1 - 0.10) * (1 + 1 / len(cal)))  # eta_hat does not depend on the level
        eps_true = float(noisy_true.mean())
        a_sorted = np.sort(-s_obs); Fr_all = random_label_cdf(cal, a_sorted)
        thr = np.empty((len(DELTAS), len(METHODS)))
        for d_i, delta in enumerate(DELTAS):
            lvl = (1 - delta) * (1 + 1 / len(cal))
            thr[d_i] = [-np.quantile(-s_clean, min(1, lvl)),
                        -np.quantile(-s_obs, min(1, lvl)),
                        -wquantile(-s_obs, W, 1 - delta),
                        (-np.quantile(a_keep, min(1, (1 - delta) * (1 + 1 / len(a_keep)))) if len(a_keep) else -np.inf),
                        -wquantile(-s_obs, p1, 1 - delta),
                        -wquantile(-s_obs, p4n, 1 - delta),
                        -nacp_threshold(a_sorted, Fr_all, n, eps_true, delta),
                        -nacp_threshold(a_sorted, Fr_all, n, e5, delta)]
        tv = thr.ravel()
        sizes = set_sizes(test, tv)
        cov = (S[test, test][:, None] >= tv[None, :]).mean(0)
        if args.check and sp_i == 0:
            ref_sizes = np.stack([(S[test] >= t).sum(1) for t in tv], 1)
            cnt_ = np.zeros(len(a_sorted) + 1, np.int64)
            for s in range(0, len(cal), args.chunk):
                cnt_ += np.bincount(np.searchsorted(a_sorted, -S[cal[s:s + args.chunk]].ravel(), side='left'),
                                    minlength=len(a_sorted) + 1)
            Fr_ref = np.cumsum(cnt_)[:len(a_sorted)] / (len(cal) * n)
            print(f'check: set-size max diff {np.abs(sizes - ref_sizes).max()}, F_r max diff {np.abs(Fr_all - Fr_ref).max():.2e}, '
                  f'top/hard5 rows equal {all(np.array_equal(np.argpartition(-S[i], kk + 1)[:kk + 1], top_s[kk][i]) for i in cal[:200])}/'
                  f'{all(np.array_equal(np.argpartition(-np.where(np.arange(n) == i, -np.inf, S[i]), mm)[:mm], hard[mm][i]) for i in cal[:200])}',
                  flush=True)
        shp = (len(DELTAS), len(METHODS))
        runs.append(dict(true_eta=eps_true, eta_hat=e4n, eta_hat_1view=e1, eta_hat_mpe=e5,
                         beta=[float(b) for b in b4n], thr=thr.tolist(), cov=cov.reshape(shp).tolist(),
                         size_mean=sizes.mean(0).reshape(shp).tolist(),
                         size_median=np.median(sizes, 0).reshape(shp).tolist(),
                         singleton=(sizes == 1).mean(0).reshape(shp).tolist(),
                         empty=(sizes == 0).mean(0).reshape(shp).tolist()))
    c = np.array([r['cov'] for r in runs]).mean(0); z = np.array([r['size_mean'] for r in runs]).mean(0)
    d0 = DELTAS.index(0.10)
    print(f'[{time.time()-t0:.0f}s] {kind} eta={eta} ncal={ncal} B={nbins} k={kk} pairs={npairs} m={mm}: '
          f'eta^ {np.mean([r["eta_hat"] for r in runs]):.3f} '
          f'(true {np.mean([r["true_eta"] for r in runs]):.3f}) | @0.90 cov naive {c[d0, 1]:.3f} NACP {c[d0, 6]:.3f} '
          f'NIRA {c[d0, 5]:.3f} | size oracle {z[d0, 0]:.1f} naive {z[d0, 1]:.1f} NIRA {z[d0, 5]:.1f}', flush=True)
    results.append(dict(noise=kind, eta=eta, ncal=ncal, bins=nbins, k=kk, pairs=npairs, hard_m=mm, splits=runs))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(dict(export=os.path.basename(args.export), split=args.split, cal_labels=os.path.basename(args.cal_labels),
               n=n, hits1=hits1, seed=args.seed, deltas=DELTAS, methods=METHODS, configs=results),
          open(args.out, 'w', encoding='utf-8'))
print(f'done in {time.time()-t0:.0f}s ->', args.out)
