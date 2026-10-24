# FB15K-DB15K protocol (frozen 2026-10-23, before any FB15K-DB15K training or analysis)

Builds on [EXTENDED_PROTOCOL.md](EXTENDED_PROTOCOL.md) (including Amendment A) and the second-base-model run
([BASE_MODEL_REPORT_zh.md](BASE_MODEL_REPORT_zh.md)). Everything not stated here is unchanged: analysis code and methods
(M0–M6, M4n, M4w, ROCP fixed-set decision layer), 20 calibration splits with analysis seed 20261022, n_cal = 3000,
target coverage 0.90, c_w = 5, c_r = 1.

## 1. Data
- FB15K-DB15K (`FBDB15K/norm`) from the PMF data package already on disk (MMKG version): 14,951 FB15K entities,
  12,842 DB15K entities, 12,846 aligned pairs, 592,213 + 99,028 triples, attribute files as shipped.
- Image features: `FBDB15K_id_img_feature_dict.pkl` from the same package (4096-d). FB 13,444 / 14,951 and
  DB 12,837 / 12,842 entities have one; 12,841 of the 12,846 aligned pairs have both.
- Split: `np.random.seed(3408); np.random.shuffle(pairs)`; the first 20% (2,569) are training seeds and the other
  10,277 are reference pairs. 20% is the standard seed ratio for this dataset and the ratio in PMF's own FB script
  (DBP15K used 30%).
- Entity names, used **only** for the second view and the LLM prompts (never given to RULE or PMF; use_surface = 0):
  - FB15K: `FB15k_mid2name.txt` from the KG-BERT repository (yao8839836/kg-bert, `data/FB15K/`, git blob 9d4cd44,
    sha256 4da94b80…). It covers all 14,951 FB15K entities.
  - DB15K: decoded from the DBpedia URIs.
  - Tokens for the view: lower case, every non-alphanumeric character replaced by a space, split on whitespace.
    Display names for the LLM: underscores replaced by spaces.
  - On this dataset 74.8% of the aligned pairs have identical normalised names, so the name view is far stronger
    than on DBP15K. This was measured on the gold alignment before any experiment and is stated here as a known
    property of the data, not a result.

## 2. Base models and runs
- RULE (the memory-light implementation checked against the original), trained with its DBP15K hyperparameters
  (40 epochs, warm-up 10, re-weighting until epoch 25, batch 512), because RULE ships no FB15K-DB15K configuration.
- PMF with `--enable_sota --il 0`, which for FB datasets sets PMF's own values (250 epochs, freeze_epochs 150);
  last-epoch model, no test-set model selection, CSLS k = 3, no surface names. Same patches as in the base-model run.
- Conditions: synthetic noise η ∈ {0.2, 0.5} (RULE-style injection, the same code as for DBP15K) and LLM-labelled
  training seeds (noise_ratio 0). Seeds 3408 / 1234 / 2025. 2 models × 3 conditions × 3 seeds = 18 trainings.
- Before the queue, each model is run once for at most 3 epochs to measure memory and time. These smoke-test outputs
  are deleted and never analysed.

## 3. LLM annotation
- Same annotator (Qwen3-4B-Instruct-2507 Q4_K_M, llama.cpp b9637), greedy, grammar-constrained to one option letter,
  K = 10 candidates = top-10 targets by the name view, shown in a seeded random order, no abstention.
- Prompt adapted to the dataset (the DBP15K prompt spoke of Wikipedia language editions):
  "Task: entity alignment between Freebase and DBpedia. Pick the DBpedia entity that refers to the same real-world
  entity as the source entity. Answer with the letter of the best option only." followed by
  "Source entity (Freebase): <FB name>" and the ten options.
- Annotated sets: all 2,569 training seeds (candidates among the training-seed targets) and a fixed random pool of
  5,000 reference pairs (seed 20261023; candidates among the 10,277 reference targets).
- With LLM calibration labels, test = the 7,277 reference pairs outside each calibration draw (DBP15K: 7,500).

## 4. Analysis and criteria
- `analyze_ext.py` with one change: a `--names` option that points the second view at the FB15K-DB15K name tokens.
  The DBP15K default path is unchanged.
- Replication criterion (EXTENDED_PROTOCOL.md §1), applied to the 12 synthetic runs (2 models × 2 η × 3 seeds):
  M4 and M4n each have coverage ≥ 0.89, |η̂ − η| ≤ 0.10 and a smaller mean set than naive CP in every run.
- LLM-noise results are reported, not pass/fail, as for DBP15K. Stated in advance: with near-identical names the
  LLM error rate is expected to be low. If the realised error rate of the calibration labels is below 3%, the LLM
  condition will be reported as a low-noise check (does a method stay at 0.90 without over-correcting?) rather than
  as evidence of noise correction.
- The decision-layer metrics are reported as for DBP15K.
