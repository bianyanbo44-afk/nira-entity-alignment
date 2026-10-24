# Extended experiments protocol (frozen 2026-10-23, before any extended result was computed)

Builds on [PILOT_PROTOCOL.md](PILOT_PROTOCOL.md); everything not stated here is unchanged (analysis code,
methods M0–M6, 20 calibration splits with analysis seed 20261022, n_cal = 3000, target coverage 0.90, kill criteria).

## 1. Synthetic-noise replication
- Language splits: zh_en, ja_en, fr_en (PMF split rule, shuffle seed 3408; 4500 training seeds, 10500 reference pairs).
- RULE seeds 3408 (pilot runs reused for zh_en), 1234, 2025; η_train ∈ {0.2, 0.5}; 18 runs in total.
- Per (split, η): mean ± sd over the 3 RULE seeds of the per-run means.
- Replication holds if, in every one of the 18 runs, M4 coverage ≥ 0.89, |η̂ − η| ≤ 0.10 and M4's mean set is smaller
  than naive CP's.

## 2. LLM-annotated seeds (real annotation noise)
- Annotator: Qwen3-4B-Instruct-2507 (Q4_K_M GGUF, local copy) via llama.cpp b9637 (Vulkan), greedy, output constrained
  by a grammar to one option letter.
- Prompt: the source entity's original-language name (decoded DBpedia URI) and K = 10 English candidate names (decoded
  URIs). Candidates = top-10 targets by the name view (char-bigram cosine of the translated source name vs the target
  name), shown in a seeded random order. The model must choose one (no abstention).
- Candidate pools: training seeds → the 4500 sup_pairs targets; calibration → the 10500 reference targets (the same
  sets the synthetic noise permutes within).
- Annotated sets: all 4500 training seeds and a fixed random pool of 5000 reference pairs (seed 20261023).
- RULE is trained on the LLM-labelled training seeds with noise_ratio = 0 (no synthetic noise added), seeds 3408 / 1234
  / 2025, all three splits.
- Conformal: per split, 3000 calibration pairs drawn from the annotated pool (labels = LLM choices); test = all other
  reference pairs (7500) with gold labels. The realized LLM error rate is measured but never given to the methods
  (except reference M6).
- Primary method unchanged (M4). Pre-registered variants, reported for synthetic runs too:
  - **M4n**: M4 with a third noise component, name-hard pairs (top-5 targets by v, excluding the observed one).
  - **M4w**: M4 posterior × inverse estimated probability that a calibration label is clean given label-free
    difficulty features z(e) = [max_t s(e,t), top-1 minus top-2 of s(e,·), max_t v(e,t)]; propensity from a fractional
    logistic regression of the M4 posteriors on z; weights normalised and clipped at the 1st/99th percentiles.
- Reported (not a kill criterion): coverage and set size of M4, M4n, M4w against naive CP and RULE weighting.

## 3. ROCP decision layer
- Actions: auto-merge the top-1 candidate, or send to review. Loss ℓ(merge t, y) = c_w·1[t ≠ y], ℓ(review, y) = c_r,
  with c_w = 5, c_r = 1.
- p(y | e) = softmax over the pool of cos(joint(e), joint(y)) / 0.07 (RULE's temperature), cosine on L2-normalised
  joint embeddings.
- ROCP with a global β (vendor/rocp.py, Algorithm 1). Under this loss the ROCP set is {top-1} (merge) or the whole pool
  (review): merge iff p1 ≥ 1 − c_r / (c_w + β); β is the smallest value whose (weighted) calibration coverage is
  ≥ 1 − α_d, α_d ∈ {0.05, 0.02}. A calibration pair is uncovered iff it is merged and its top-1 differs from its label.
  This closed form is checked against vendor/rocp.py on a small label space before use.
- Calibration variants: oracle (gold labels), naive (noisy labels, uniform), RULE weight, M4 posterior, M4w.
- Test metrics: merge rate, error rate among merges, overall wrong-merge rate (target ≤ α_d), review rate, mean loss.

### Amendment A (2026-10-23, before any comparison between methods existed)
The regression run on the pilot export (zh_en, η = 0.2) showed the global-β ROCP set above is degenerate here: a
softmax at temperature 0.07 over 10,500 candidates is so flat that only 3.5% of entities clear ROCP's own floor
p1 ≥ 1 − c_r/c_w = 0.8, so every calibration variant merged the same 3.5% with zero errors and the α_d constraint never
bound. That output is kept and reported, but it cannot compare methods.

Added as the decision layer used for comparison: ROCP's minimax-optimal policy for a **fixed** prediction set
(vendor/rocp.py `action_and_certificate`) applied to each method's conformal set at α = 0.10. With this loss it is
"merge t iff C = {t} and α·c_w ≤ c_r, otherwise review" (checked against the vendor code: 0 mismatches in 1,200 random
sets; the strict-inequality version differs only at the tie α·c_w = c_r). It needs no extra probability model.
Metrics per method: merge rate, wrong-merge rate (overall), error among merges, mean loss (c_w per wrong merge, c_r per
review). The global-β closed form matched vendor/rocp.py in 1,500 of 1,500 random cases.
