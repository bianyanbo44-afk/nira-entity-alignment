# Minimum pilot protocol (frozen 2026-10-22, before any real RULE output was seen)

## Models
- RULE (ICLR 2026) on DBP15K zh_en, 30% seeds (4500) / 70% reference pairs (10500), split by
  `np.random.seed(3408); shuffle(ill_ent_ids)` (PMF convention; RULE's own Baidu split was not available).
- Two runs: training-seed noise η_train ∈ {0.2, 0.5}, RULE defaults (40 epochs, seed 3408, use_surface=0,
  so entity names are NOT used by RULE). RULE injects its own noise: η of the seed pairs get their targets
  permuted, and image features of η of the seed targets are shuffled.
- Code: released RULE + bytecode-recovered `gcn_layer.py` + memory-only patches (chunked DRF_test checked
  against the original, sparse relation counts, no DataLoader workers). Patches are marked `[pilot patch]`.
- Image features: PMF's clean `zh_en_GA_id_img_feature_dict.pkl` (RULE's own noisy image pkls are only on Baidu).

## Conformal setting
- Pool: the 10500 right-side entities of the reference pairs (RULE's own test candidate space).
- 20 random splits of the reference pairs: n_cal = 3000 calibration, n_test = 7500 test.
- Calibration labels are corrupted at η_cal = η_train with RULE's mechanism (permute the targets of a random
  η fraction of calibration pairs). Test labels stay gold. Target coverage 1 − δ = 0.90.
- Primary score s(e,t) = joint(e)·joint(t) (RULE's similarity before CSLS); C(e) = {t in pool : s(e,t) ≥ τ}.
- Second view v(e,t) = cosine of character-bigram TF vectors of the translated entity names.

## Methods
| id | method |
|---|---|
| M0 | oracle: split CP on gold calibration labels (reference only) |
| M1 | naive split CP on noisy calibration labels |
| M2 | RULE reliability weighting: W = ½(1−u) + ½·clamp(cos⁺,0,1), RULE's EDL uncertainty (τ=0.07, top-49 in-batch negatives, batches of 512), averaged over both directions; weighted CP |
| M2b | RULE filter: drop calibration pairs with W < 0.5, split CP on the rest |
| M3 | 1-view EM posterior weighting (score only; noise kernel = β·random + (1−β)·near-neighbour pairs) |
| M4 | **2-view EM posterior weighting (proposed)**: (s, v) conditionally independent given clean/noisy |
| M5 | single-view MPE deconvolution: η̂ = min ratio of observed vs random-pair survival on the low-similarity side (random survival ≥ 0.2), deconvolved CDF quantile |
| M6 | deconvolution with the true η (reference only) |

Secondary analyses (reported, not used for the kill decision): RULE's 2-iteration CSLS score instead of the raw
dot product; "hard" calibration noise (target replaced by one of the 5 most similar wrong targets); n_cal = 1000.

## Metrics
Test coverage, mean and median set size, singleton rate, η̂ (EM, MPE). Mean ± sd over the 20 splits.

## Kill criteria (primary setting: dot score, random noise, n_cal = 3000; both η)
- **K1 problem not real**: naive CP coverage ≤ 0.92 AND naive mean set size ≤ 1.5 × oracle, at BOTH η.
- **K2 method does not help**: at EITHER η, M4 has mean coverage < 0.89, OR M4's mean set size ≥ naive's,
  OR RULE weighting (M2) is valid (coverage ≥ 0.89) with a smaller mean set than M4.
- **K3 identifiability fails**: |mean η̂(M4) − η| > 0.10 at EITHER η.
Any of K1–K3 → stop or re-scope the idea.

Claim check (not a kill criterion): if a single-view method (M3 or M5) is also valid with a mean set within
10% of M4's at both η, the "second view is needed" claim does not hold for RULE-style random noise.

## Amendment (2026-10-22, before any RULE result existed): execution environment only
The laptop ran out of Windows commit memory, so the runs move to a rented Linux RTX 5090 (32 GB), PyTorch 2.8 /
CUDA 12.8, via `run_cloud.sh`. There RULE uses its original code path (`RULE_LEAN=0`: torch.pca_lowrank on the
materialised edge matrix, no checkpointing, original DRF_test). Remaining differences from the release are the
bytecode-recovered `gcn_layer.py`, the export hook, `workers=0`, and loader/fusion patches that are exactly
row-wise or count-equivalent. The chunked DRF_test is still checked against the original on all 10500 links.
Methods, splits, metrics and kill criteria above are unchanged.

## Limitations fixed in advance
One training seed per η; calibration variability only. Names are clean (RULE does not add name noise when
use_surface=0). The split and the image pkl differ from RULE's private benchmark files, so absolute Hits@1 may
differ from the paper.
