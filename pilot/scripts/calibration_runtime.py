"""Equivalent cached computations for the calibration reanalysis.

Bin membership and sorted score rows are deterministic and reused. No model,
sampling, stopping rule, iteration count or calibration threshold is changed.
"""
import numpy as np

CONTEXT = {}
VERIFY = False
VERIFY_ALL = False
_checked = set()
_matrix = None
_sorted = None


def _bins(x, edges):
    return np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)


def _hist(indices, size, weights=None):
    h = np.bincount(indices, weights=weights, minlength=size) + 1e-3
    return h / h.sum()


def em(s, v, sr, vr, sh, vh, use_view, iters=300):
    edges, hist, look = (CONTEXT[x] for x in ('edges', 'hist', 'look'))
    es = edges(np.concatenate([s, sr, sh])); ev = edges(np.concatenate([v, vr, vh]))
    si, vi = _bins(s, es), _bins(v, ev)
    fr, fh = hist(sr, es), hist(sh, es); gr, gh = hist(vr, ev), hist(vh, ev)
    ar = fr[si] * (gr[vi] if use_view else 1)
    ah = fh[si] * (gh[vi] if use_view else 1)
    p = np.full(len(s), .8); beta, eta = .5, .2
    for _ in range(iters):
        fc = _hist(si, len(es) - 1, p); gc = _hist(vi, len(ev) - 1, p)
        lc = (1 - eta) * fc[si] * (gc[vi] if use_view else 1)
        mix = beta * ar + (1 - beta) * ah
        p = lc / (lc + eta * mix)
        resp = (1 - p) * beta * ar / mix
        eta = float(np.clip(1 - p.mean(), 1e-3, .95))
        beta = float(np.clip(resp.sum() / max((1 - p).sum(), 1e-9), .01, .99))
    return p, eta, beta


def em_multi(s, v, kernels, iters=300, nb=40):
    edges, hist = (CONTEXT[x] for x in ('edges', 'hist'))
    es = edges(np.concatenate([s] + [k[0] for k in kernels]), nb)
    ev = edges(np.concatenate([v] + [k[1] for k in kernels]), nb)
    si, vi = _bins(s, es), _bins(v, ev)
    a = np.stack([hist(ks, es)[si] * hist(kv, ev)[vi] for ks, kv in kernels])
    p = np.full(len(s), .8); beta = np.full(len(kernels), 1 / len(kernels)); eta = .2
    for _ in range(iters):
        fc = _hist(si, len(es) - 1, p); gc = _hist(vi, len(ev) - 1, p)
        mix = (beta[:, None] * a).sum(0)
        lc = (1 - eta) * fc[si] * gc[vi]
        p = lc / (lc + eta * mix)
        resp = (1 - p)[None, :] * beta[:, None] * a / mix[None, :]
        eta = float(np.clip(1 - p.mean(), 1e-3, .95))
        beta = np.clip(resp.sum(1) / max((1 - p).sum(), 1e-9), .01, .99)
        beta /= beta.sum()
    return p, eta, beta


def sorted_matrix(matrix):
    global _matrix, _sorted
    if matrix is not _matrix:
        _matrix, _sorted = None, None
        _sorted = np.sort(matrix, axis=1)
        _matrix = matrix
    return _sorted


def random_label_cdf(matrix, cal, grid):
    rows = sorted_matrix(matrix); n = matrix.shape[1]
    counts = np.zeros(len(grid), np.int64)
    for i in cal:
        counts += n - np.searchsorted(rows[i], -grid, side='left')
    return counts / (len(cal) * n)


def evaluate_many(matrix, test, thresholds):
    rows = sorted_matrix(matrix); n = matrix.shape[1]
    keys = list(thresholds); values = np.array([thresholds[k] for k in keys])
    sizes = np.stack([n - np.searchsorted(rows[i], values, side='left') for i in test])
    true = np.asarray(matrix[test, test]); out = {}
    for j, name in enumerate(keys):
        z = sizes[:, j]; covered = true >= thresholds[name]
        merge = (z == 1) & (CONTEXT['args'].delta * CONTEXT['C_W'] <= CONTEXT['C_R'])
        wrong = merge & ~covered
        out[name] = dict(cov=float(covered.mean()), size_mean=float(z.mean()),
                         size_median=float(np.median(z)), singleton=float((z == 1).mean()),
                         empty=float((z == 0).mean()), thr=float(thresholds[name]),
                         dec_merge=float(merge.mean()), dec_wrong=float(wrong.mean()),
                         dec_merge_err=float(wrong.sum() / max(merge.sum(), 1)),
                         dec_loss=float((CONTEXT['C_W'] * wrong + CONTEXT['C_R'] * ~merge).mean()))
    return out


def wrap(reference, name):
    optimized = globals()[name]
    if not VERIFY:
        return optimized

    def checked(*args, **kwargs):
        result = optimized(*args, **kwargs)
        signature = (name, kwargs.get('use_view'), kwargs.get('nb', 40))
        if VERIFY_ALL or signature not in _checked:
            expected = reference(*args, **kwargs)
            def compare(a, b):
                if isinstance(a, dict):
                    for key in a: compare(a[key], b[key])
                elif isinstance(a, tuple):
                    for x, y in zip(a, b): compare(x, y)
                else:
                    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9, equal_nan=True)
            compare(result, expected)
            _checked.add(signature)
            print('Equivalent computation verified:', name, flush=True)
        return result
    return checked
