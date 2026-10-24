# Extended post-hoc conformal analysis (EXTENDED_PROTOCOL.md). M0–M6 are the pilot's methods with the pilot's code and
# random-draw order (analyze_cp.py), so synthetic runs reproduce the pilot exactly; the additions (M4n, M4w, ROCP
# decision layer, LLM calibration labels, other language splits) draw no random numbers.
# usage: python -I analyze_ext.py <export.npz> <data_dir> <out_prefix> <eta> [--split zh_en] [--cal_labels annot.jsonl]
import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('export'); ap.add_argument('data_dir'); ap.add_argument('out_prefix'); ap.add_argument('eta', type=float)
ap.add_argument('--splits', type=int, default=20); ap.add_argument('--ncal', type=int, default=3000)
ap.add_argument('--delta', type=float, default=0.10); ap.add_argument('--seed', type=int, default=20261022)
ap.add_argument('--chunk', type=int, default=1000)
ap.add_argument('--split', default='zh_en'); ap.add_argument('--cal_labels', default='')
ap.add_argument('--keep_matrices', action='store_true')
ap.add_argument('--names', default='', help='name-token file for the second view (default: DBP15K translated names)')
args = ap.parse_args()
C_W, C_R, TAU_SOFTMAX, ALPHAS_D = 5.0, 1.0, 0.07, (0.05, 0.02)
t0 = time.time()
E = np.load(args.export)
ref = E['test_ill'].astype(np.int64)            # (10500, 2) gold reference pairs
ent_idx = E['ent_idx'].astype(np.int64)
pos = {int(e): i for i, e in enumerate(ent_idx)}
J = E['joint'].astype(np.float32)
L = J[[pos[int(e)] for e in ref[:, 0]]]
R = J[[pos[int(e)] for e in ref[:, 1]]]
n = len(ref)


# ---- score matrices as disk-backed memmaps (rows: ref sources, cols: ref targets = pool) ----
def memmap(name):
    return np.lib.format.open_memmap(f'{args.out_prefix}_{name}.npy', mode='w+', dtype=np.float32, shape=(n, n))


S = memmap('dot')
for s in range(0, n, args.chunk):
    S[s:s + args.chunk] = L[s:s + args.chunk] @ R.T
S.flush()


def topk_mean_rows_cols(M, k=3, a=None, b=None, scale=1.0):
    """row/col mean of top-k of (scale*M - a[:,None] - b[None,:]) computed chunkwise."""
    rows = np.empty(n, np.float32); col_top = np.full((k, n), -np.inf, np.float32)
    for s in range(0, n, args.chunk):
        X = scale * np.asarray(M[s:s + args.chunk])
        if a is not None: X = X - a[s:s + args.chunk, None] - b[None, :]
        rows[s:s + args.chunk] = -np.sort(-np.partition(X, -k, axis=1)[:, -k:], axis=1).mean(1)
        cand = np.vstack([col_top, np.partition(X, -k, axis=0)[-k:]])
        col_top = -np.sort(-cand, axis=0)[:k]
    return rows, col_top.mean(0)


# RULE csls_sim with k=3, m_csls=2: C1 = 2S - r1 - c1 ; C2 = 2C1 - r2 - c2
r1, c1 = topk_mean_rows_cols(S)
r2, c2 = topk_mean_rows_cols(S, a=r1, b=c1, scale=2.0)
C = memmap('csls')
for s in range(0, n, args.chunk):
    C1 = 2 * np.asarray(S[s:s + args.chunk]) - r1[s:s + args.chunk, None] - c1[None, :]
    C[s:s + args.chunk] = 2 * C1 - r2[s:s + args.chunk, None] - c2[None, :]
C.flush()
hits1 = {k: float(np.mean([np.argmax(M[i]) == i for i in range(n)])) for k, M in [('dot', S), ('csls', C)]}
print(f'[{time.time()-t0:.0f}s] Hits@1 over {n}x{n}: {hits1}', flush=True)

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

# ---- label-free per-source quantities (no random draws) ----
# name view: row max and top-6 neighbours (for M4n's name-hard kernel and M4w's features)
vmax = np.empty(n, np.float32); vtop = np.empty((n, 6), np.int64)
NRt = NR.T.tocsc()
for s in range(0, n, args.chunk):
    X = (NL[s:s + args.chunk] @ NRt).toarray()
    vmax[s:s + args.chunk] = X.max(1)
    vtop[s:s + args.chunk] = np.argpartition(-X, 6, axis=1)[:, :6]
# ROCP: softmax over the pool of cos / 0.07 -> top-1 index and its probability
p1_all = np.empty(n, np.float64); top1_all = np.empty(n, np.int64)
for s in range(0, n, args.chunk):
    X = (Ln[s:s + args.chunk] @ Rn.T).astype(np.float64) / TAU_SOFTMAX
    mx = X.max(1, keepdims=True)
    p1_all[s:s + args.chunk] = 1.0 / np.exp(X - mx).sum(1)
    top1_all[s:s + args.chunk] = X.argmax(1)
# score-matrix row statistics (M4w features)
rowstat = {}
for name_, M in [('dot', S), ('csls', C)]:
    mx, mg = np.empty(n, np.float32), np.empty(n, np.float32)
    for s in range(0, n, args.chunk):
        X = np.array(M[s:s + args.chunk]); t2 = np.partition(X, -2, axis=1)[:, -2:]
        mx[s:s + args.chunk] = t2.max(1); mg[s:s + args.chunk] = t2.max(1) - t2.min(1)
    rowstat[name_] = (mx, mg)
print(f'[{time.time()-t0:.0f}s] label-free statistics ready (ROCP top-1 acc {np.mean(top1_all == np.arange(n)):.4f})', flush=True)


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
    """smallest a* with sum_{a_i<=a*} w_i / (sum w + max w) >= level ; +inf if never."""
    if not np.any(w > 0):
        return np.inf
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


def em_multi(s, v, kernels, iters=300):
    """M4n: two-view EM with several noise components (list of (s_samples, v_samples))."""
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


def frac_logit(Z, y, l2=1e-3, iters=60):
    """fractional logistic regression of y in [0,1] on Z (IRLS); returns fitted probabilities."""
    X = np.hstack([np.ones((len(Z), 1)), Z]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); Wt = q * (1 - q) + 1e-9
        step = np.linalg.solve((X * Wt[:, None]).T @ X + l2 * np.eye(X.shape[1]), X.T @ (y - q) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-8: break
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


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


def random_label_cdf(M, cal, grid):
    """F_r(q) = mean_i |C_q(x_i)| / k on a sorted grid of nonconformity values, exact over all k candidates of the
    calibration sources (the coverage of a uniformly random label)."""
    cnt = np.zeros(len(grid) + 1, np.int64)
    for s in range(0, len(cal), args.chunk):
        A = -np.asarray(M[cal[s:s + args.chunk]], np.float32).ravel()
        cnt += np.bincount(np.searchsorted(grid, A, side='left'), minlength=len(grid) + 1)
    return np.cumsum(cnt)[:len(grid)] / (len(cal) * M.shape[1])


def nacp_threshold(a_sorted, Fr_all, k, eps, delta):
    """NACP (Penso, Goldberger and Fetaya, COPA 2025, Algorithm 1), uniform label noise with rate eps over the k candidates.
    Clean coverage estimate F_c(q) = (F_n(q) - eps * F_r(q)) / (1 - eps), with F_n the empirical CDF of the noisy-label
    nonconformity a_sorted and F_r = random_label_cdf on the grid a_sorted. Grid search over [q1, q2] (their Theorem 2
    bounds) for F_c(q) = 1 - delta, largest solution. No random draws."""
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
    up = np.where(ok[1:] & ~ok[:-1])[0] + 1  # up-crossings of the target; the largest one is NACP's solution
    return grid[up[-1]] if len(up) else grid[int(np.argmax(ok))]


def evaluate_many(M, test, thrs):
    """thrs: {method: similarity threshold}; C(e) = {t : M[e,t] >= thr}. One pass over the test rows.
    dec_*: ROCP's minimax policy for a fixed set (vendor action_and_certificate, checked in check_rocp.py): with
    alpha = delta, merge iff C = {t} and alpha*c_w <= c_r, otherwise review (empty sets are reviewed too)."""
    true_s = np.asarray(M[test, test])
    sizes = {m: np.empty(len(test), np.int32) for m in thrs}
    for s in range(0, len(test), args.chunk):
        X = np.array(M[test[s:s + args.chunk]])
        for m, t in thrs.items():
            sizes[m][s:s + args.chunk] = (X >= t).sum(1)
    out = {}
    for m, t in thrs.items():
        covered = true_s >= t
        merge = (sizes[m] == 1) & (args.delta * C_W <= C_R)
        wrong = merge & ~covered
        out[m] = dict(cov=float(np.mean(covered)), size_mean=float(sizes[m].mean()),
                      size_median=float(np.median(sizes[m])), singleton=float(np.mean(sizes[m] == 1)),
                      empty=float(np.mean(sizes[m] == 0)), thr=float(t),
                      dec_merge=float(merge.mean()), dec_wrong=float(wrong.mean()),
                      dec_merge_err=float(wrong.sum() / max(merge.sum(), 1)),
                      dec_loss=float((C_W * wrong + C_R * ~merge).mean()))
    return out


def rocp_threshold(p1c, err, w, alpha):
    """ROCP with the merge/review loss: merge iff p1 >= 1 - c_r/(c_w + beta), beta the smallest value whose weighted
    calibration coverage is >= 1 - alpha. Equivalent direct form: the lowest p1 threshold (>= 1 - c_r/c_w) at which
    the weighted share of merged-and-wrong calibration pairs is <= alpha."""
    o = np.argsort(-p1c, kind='mergesort'); ps = p1c[o]
    ce = np.cumsum(w[o] * err[o]) / w.sum()
    ok = np.nonzero(ce <= alpha)[0]
    if len(ok) == 0 or ok[0] != 0:
        return np.inf
    kmax = ok[-1] if np.all(np.diff(ok) == 1) else np.nonzero(np.diff(ok) != 1)[0][0]
    tau = ps[kmax]
    if kmax + 1 < len(ps) and ps[kmax + 1] == tau:
        tau = np.nextafter(tau, np.inf)
    return max(tau, 1 - C_R / C_W)


def rocp_eval(cal, labels, w, test):
    out = {}
    err = (top1_all[cal] != labels).astype(np.float64)
    for a_d in ALPHAS_D:
        tau = rocp_threshold(p1_all[cal], err, np.asarray(w, np.float64), a_d)
        merged = p1_all[test] >= tau
        wrong = merged & (top1_all[test] != test)
        out[str(a_d)] = dict(tau=float(tau), merge_rate=float(merged.mean()),
                             merge_err=float(wrong.sum() / max(merged.sum(), 1)), wrong_overall=float(wrong.mean()),
                             review_rate=float(1 - merged.mean()), mean_loss=float((C_W * wrong + C_R * ~merged).mean()))
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

settings = ([('dot', S, 'llm'), ('csls', C, 'llm')] if args.cal_labels else
            [('dot', S, 'random'), ('dot', S, 'hard'), ('csls', C, 'random')])
rng = np.random.default_rng(args.seed)
results = {}
for score_name, M, noise_kind in settings:
    key = f'{score_name}|{noise_kind}|ncal{args.ncal}'
    runs = []
    for sp_i in range(args.splits):
        if noise_kind == 'llm':
            perm = rng.permutation(pool_idx); cal = perm[:args.ncal]
            test = np.setdiff1d(np.arange(n), cal)
            obs = np.array([llm_obs[int(i)] for i in cal], np.int64)
        else:
            perm = rng.permutation(n); cal, test = perm[:args.ncal], perm[args.ncal:]
            obs = cal.copy()
            k_noisy = int(len(cal) * args.eta)
            sel = rng.choice(len(cal), k_noisy, replace=False)
            if noise_kind == 'random':          # RULE's mechanism: permute targets among selected pairs
                obs[sel] = obs[sel][rng.permutation(k_noisy)]
            else:                               # hard: one of the 5 most similar wrong targets
                for kq in sel:
                    i = cal[kq]
                    row = np.array(M[i]); row[i] = -np.inf          # copy: never write into the memmap
                    obs[kq] = rng.choice(np.argpartition(-row, 5)[:5])
        noisy_true = obs != cal
        s_obs = np.asarray(M[cal, obs], np.float32)
        s_clean = np.asarray(M[cal, cal], np.float32)
        v_obs = vsim(cal, obs)
        # negative pools (observable): random pairs and near-neighbour pairs of calibration sources
        # Sample without consulting gold correspondences. A random reference pair may be a true match.
        ri = rng.choice(cal, 20000); rj = rng.integers(0, n, 20000)
        s_r = np.asarray(M[ri, rj], np.float32); v_r = vsim(ri, rj)
        hi, hj = [], []
        for kq, i in enumerate(cal):
            row = np.array(M[i]); top = np.argpartition(-row, 6)[:6]
            top = [t for t in top if t != obs[kq]][:5]
            hi += [i] * len(top); hj += top
        hi, hj = np.array(hi), np.array(hj)
        s_h = np.asarray(M[hi, hj], np.float32); v_h = vsim(hi, hj)
        lvl = (1 - args.delta) * (1 + 1 / len(cal))
        W = rule_weight(cal, obs, rng)
        keep = W >= 0.5
        p1, e1, b1 = em(s_obs, v_obs, s_r, v_r, s_h, v_h, use_view=False)
        p2, e2, b2 = em(s_obs, v_obs, s_r, v_r, s_h, v_h, use_view=True)
        q5, e5 = mpe_deconv(-s_obs, -s_r, lvl)
        q6, _ = mpe_deconv(-s_obs, -s_r, lvl, eta_fixed=float(noisy_true.mean()))
        a_keep = -s_obs[keep]
        # --- additions (no random draws) ---
        vi = np.repeat(cal, 5)
        vj = np.array([[t for t in vtop[i] if t != obs[kq]][:5] for kq, i in enumerate(cal)]).ravel()
        s_vh = np.asarray(M[vi, vj], np.float32); v_vh = vsim(vi, vj)
        p4n, e4n, b4n = em_multi(s_obs, v_obs, [(s_r, v_r), (s_h, v_h), (s_vh, v_vh)])
        Z = np.column_stack([rowstat[score_name][0][cal], rowstat[score_name][1][cal], vmax[cal]]).astype(np.float64)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        pi_hat = frac_logit(Z, p2.astype(np.float64))
        w4w = p2 / np.maximum(pi_hat, 1e-6)
        lo_, hi_ = np.quantile(w4w, [0.01, 0.99]); w4w = np.clip(w4w, lo_, hi_)
        # published uniform-noise baseline (NACP) with the realised noise rate and with the MPE estimate; no random draws
        eps_true = float(noisy_true.mean())
        a_sorted = np.sort(-s_obs); Fr_all = random_label_cdf(M, cal, a_sorted)
        q_nacp_true = nacp_threshold(a_sorted, Fr_all, n, eps_true, args.delta)
        q_nacp_mpe = nacp_threshold(a_sorted, Fr_all, n, e5, args.delta)
        thrs = {  # similarity thresholds = -(nonconformity quantile)
            'M0_oracle': -np.quantile(-s_clean, min(1, lvl)),
            'M1_naive': -np.quantile(-s_obs, min(1, lvl)),
            'M2_rule_weight': -wquantile(-s_obs, W, 1 - args.delta),
            'M2b_rule_filter': (-np.quantile(a_keep, min(1, (1 - args.delta) * (1 + 1 / len(a_keep))))
                                if len(a_keep) else -np.inf),
            'M3_em_1view': -wquantile(-s_obs, p1, 1 - args.delta),
            'M4_em_2view': -wquantile(-s_obs, p2, 1 - args.delta),
            'M5_mpe_deconv': -q5,
            'M6_deconv_true_eta': -q6,
            'M4n_em_namehard': -wquantile(-s_obs, p4n, 1 - args.delta),
            'M4w_em_ipw': -wquantile(-s_obs, w4w, 1 - args.delta),
            'B1_nacp_true_eta': -q_nacp_true,
            'B2_nacp_mpe_eta': -q_nacp_mpe,
        }
        out = evaluate_many(M, test, thrs)
        out['B1_nacp_true_eta']['eps'] = eps_true
        out['B2_nacp_mpe_eta']['eps'] = float(e5)
        out['M2b_rule_filter']['kept_frac'] = float(keep.mean())
        out['M2b_rule_filter']['kept_noisy_frac'] = float(noisy_true[keep].mean()) if keep.any() else np.nan
        out['M3_em_1view'].update(eta_hat=e1, beta=b1)
        out['M4_em_2view'].update(eta_hat=e2, beta=b2)
        out['M4_em_2view']['post_auc'] = float(((p2[~noisy_true][:, None] > p2[noisy_true][None, :]).mean())
                                               if noisy_true.any() and (~noisy_true).any() else np.nan)
        out['M2_rule_weight']['w_auc'] = float(((W[~noisy_true][:, None] > W[noisy_true][None, :]).mean())
                                               if noisy_true.any() and (~noisy_true).any() else np.nan)
        out['M5_mpe_deconv']['eta_hat'] = e5
        out['M4n_em_namehard'].update(eta_hat=e4n, beta_random=float(b4n[0]), beta_shard=float(b4n[1]),
                                      beta_vhard=float(b4n[2]))
        out['M4w_em_ipw'].update(pi_sd=float(pi_hat.std()), pi_corr_clean=float(np.corrcoef(pi_hat, ~noisy_true)[0, 1])
                                 if noisy_true.any() and (~noisy_true).any() else np.nan)
        out['_true_eta'] = float(noisy_true.mean())
        out['_v_corr_clean'] = float(np.corrcoef(s_obs[~noisy_true], v_obs[~noisy_true])[0, 1])
        out['_v_corr_noisy'] = (float(np.corrcoef(s_obs[noisy_true], v_obs[noisy_true])[0, 1])
                                if noisy_true.sum() > 2 else np.nan)
        out['_rocp'] = {'oracle': rocp_eval(cal, cal, np.ones(len(cal)), test),
                        'naive': rocp_eval(cal, obs, np.ones(len(cal)), test),
                        'rule_weight': rocp_eval(cal, obs, W, test),
                        'em_2view': rocp_eval(cal, obs, p2, test),
                        'em_ipw': rocp_eval(cal, obs, w4w, test)}
        runs.append(out)
        print(f'[{time.time()-t0:.0f}s] {key} split {sp_i}: naive cov {out["M1_naive"]["cov"]:.3f} size {out["M1_naive"]["size_mean"]:.1f} | '
              f'EM2 cov {out["M4_em_2view"]["cov"]:.3f} size {out["M4_em_2view"]["size_mean"]:.1f} eta^ {e2:.3f} | '
              f'M4w cov {out["M4w_em_ipw"]["cov"]:.3f} | ROCP@0.05 merge {out["_rocp"]["em_2view"]["0.05"]["merge_rate"]:.3f}', flush=True)
    agg = {}
    for mname in runs[0]:
        if mname == '_rocp':
            agg[mname] = {var: {a_d: {k: dict(mean=float(np.mean([r[mname][var][a_d][k] for r in runs])),
                                               sd=float(np.std([r[mname][var][a_d][k] for r in runs])))
                                      for k in runs[0][mname][var][a_d]}
                                for a_d in runs[0][mname][var]} for var in runs[0][mname]}
        elif mname.startswith('_'):
            agg[mname] = dict(mean=float(np.nanmean([r[mname] for r in runs])), sd=float(np.nanstd([r[mname] for r in runs])))
        else:
            agg[mname] = {k: dict(mean=float(np.nanmean([r[mname][k] for r in runs])),
                                  sd=float(np.nanstd([r[mname][k] for r in runs])))
                          for k in runs[0][mname]}
    results[key] = dict(agg=agg, splits=runs)
json.dump(dict(eta=args.eta, split=args.split, export=os.path.basename(args.export), cal_labels=args.cal_labels,
               hits1=hits1, rocp_top1_acc=float(np.mean(top1_all == np.arange(n))), results=results),
          open(f'{args.out_prefix}_results.json', 'w', encoding='utf-8'), indent=1)
if not args.keep_matrices:
    import gc
    settings = M = S = C = None  # drop every reference so Windows can unmap and delete the files
    gc.collect()
    for nm in ('dot', 'csls'):
        try:
            os.remove(f'{args.out_prefix}_{nm}.npy')
        except OSError:
            pass
print(f'done in {time.time()-t0:.0f}s ->', f'{args.out_prefix}_results.json')
