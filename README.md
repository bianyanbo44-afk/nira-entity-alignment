# NIRA: calibration code, annotations and numerical results

This material accompanies *When Modalities Disagree: Noise-Identifiable Conformal Calibration for Multi-Modal Entity Alignment with Noisy Seeds* by Yanbo Bian and Zhihai Wang.

## Included observations and code

- The four `annot_*.jsonl` files contain the original model annotations. Corresponding `items_*.jsonl` files preserve the source entities, candidate ordering and names shown to the annotator.
- `pilot/runs/label_free_v2/main` contains all 72 recomputed analysis files. Each model/noise analysis uses 20 calibration draws.
- `pilot/runs/label_free_v2/robust` and `hparam` contain the complete 48-file robustness sweep and 48-file hyperparameter sweep. Each configuration uses 10 calibration draws.
- `paper/figs/source_data` contains the numerical figure data. The complete numerical outputs preserve training-seed and calibration-draw information, including the error fraction among merged entities.
- Calibration programs, the reanalysis driver, the table generator and all five figure scripts are included. Existing trained models were kept fixed during this reanalysis.

The public benchmark releases, image features and trained embedding exports are not duplicated in this archive. The analysis accepts the embedding-export schema below. Base-model training follows the configurations described in the manuscript and the official RULE and PMF implementations. Training seeds are 3408, 1234 and 2025. Model-export names identify the model, benchmark, training noise and training seed.

## Numerical environment

The corrected analysis was run with Python 3.12, NumPy 2.5.3 and SciPy 1.18.1. Plotting used Matplotlib 3.11.1. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 2 to match the reanalysis runtime. Reference and cached computations are compared on the same real inputs by the `--verify` or `--verify-all` option.

## Recreate tables from the deposited numerical outputs

Extract the archive while preserving its directory structure, then run:

```text
python -I paper/make_tables.py
```

The generator requires 72 main results, 48 robustness outputs and 48 hyperparameter outputs. It reads only the corrected results and does not fall back to older numbers.

The five figures can be recreated from the included numerical outputs without the embedding exports:

```text
python -I paper/figs/fig2_mechanism.py
python -I paper/figs/fig3_calibration_map.py
python -I paper/figs/fig4_identification.py
python -I paper/figs/fig5_decisions.py
python -I paper/figs/fig6_robustness.py
```

The standalone panel-alignment auditor from the local `nature-figure` skill is bundled with these scripts. Each export must pass its alignment gate. Regenerating the mechanism observations themselves with `fig2_data.py` additionally requires the indicated frozen model export and name files.

## Recompute calibration from model exports

An export is a NumPy `.npz` file with:

- `joint`: the final fused entity-embedding matrix, with one row per `ent_idx` entry;
- `ent_idx`: entity identifiers in embedding-row order;
- `test_ill`: a two-column array of clean reference correspondences, used to define experimental corruption and measure coverage;
- any other arrays retained by the original training export are ignored by calibration.

For example, from the archive root:

```text
python -I pilot/scripts/run_calibration_fast.py --verify pilot/scripts/analyze_ext.py pilot/runs/ext/rule_zh_en_syn0.5_s3408.npz pilot/data pilot/runs/label_free_v2/main/rule/zh_en_syn0.5_s3408 0.5 --split zh_en
```

The name-token files are supplied by the benchmark releases. For FB15K--DB15K, pass the prepared English name-token mapping with `--names`. For LLM annotations, pass the corresponding `annot_*.jsonl` file with `--cal_labels`.

When all exports and benchmark name files are present, the complete analysis can be recomputed with:

```text
python -I pilot/scripts/recheck_label_free.py --workers 4
```

Existing completed outputs are reused. Recomputing a completed result requires using a fresh output directory or moving that specific result aside first. The source `analyze_ext.py` can also run directly without caching; cached functions only reuse bin membership and exact sorted-row candidate counts.

## Annotation provenance

The annotator was Qwen3-4B-Instruct-2507, using the Q4_K_M GGUF quantization and greedy decoding through llama.cpp. The prompts and decoding parameters are in `llm_annotate.py`; the candidate lists in `items_*.jsonl` are preserved without regeneration. To rerun annotation, set `--server-exe` and `--model` to local installations of llama.cpp and the stated model. No new annotation was generated during the calibration correction.

## Gold-label boundary

Synthetic corruption and evaluation use gold reference correspondences. The calibration method observes noisy seed targets, alignment scores and name similarities. Uniform reference-kernel sampling retains every sampled pair, including any accidental true matches, without consulting the clean correspondence index. The hard-kernel shortlists exclude observed seed targets, not gold targets.

## Quantile implementations

The unweighted oracle, naive and reliability-filtered reference calibrations use NumPy linear quantile interpolation at the adjusted empirical level min(1, (1-delta)(1+1/n)), with the retained sample count for filtering. NIRA uses the discrete weighted quantile written in the manuscript. These numerical conventions are preserved explicitly in the released analysis code.

## Metric denominators

`dec_merge` is the merged fraction of all test entities. `dec_wrong` is the fraction of all test entities that are merged incorrectly. `dec_merge_err` is the error fraction among merged entities, computed within each calibration draw before aggregation. The coverage bound controls the second quantity through miscoverage; it does not provide the same bound for the third quantity.

## Archive manifest

The archive includes a SHA-256 manifest for the included observations, code and numerical files. LLM server logs, local model weights, temporary score matrices and private runtime paths are excluded.


## Versioned deposit and citation

Version: **1.0.0**. Release date: **2026-10-24**.

- Repository: https://github.com/bianyanbo44-afk/nira-entity-alignment
- Versioned record: https://github.com/bianyanbo44-afk/nira-entity-alignment/releases/tag/v1.0.0
- Citation metadata: `CITATION.cff`
- Dataset-to-file mapping and access routes: `DATA_INVENTORY.json`
- Integrity: `MANIFEST.sha256.json` lists every deposited file with its SHA-256 digest.

This is a versioned GitHub deposit. No DOI has been assigned. The release download contains the same calibration observations and numerical results as the repository. Future scientific changes will receive a new version.

## Reused public sources

- [RULE official implementation](https://github.com/XLearning-SCU/2026-ICLR-RULE).
- [PMF official implementation and benchmark download instructions](https://github.com/ninibymilk/PMF-MMEA): DBP15K names, alignments and image features, and the MMKG FB15K--DB15K release used here.
- [KG-BERT name mapping](https://github.com/yao8839836/kg-bert): `FB15k_mid2name`, used to obtain Freebase entity names.

The external benchmark downloads, pretrained annotation-model weights and frozen entity embeddings are not mirrored in this deposit. The recorded annotations, candidate lists and all numerical outputs needed to recreate the manuscript tables and five figures are deposited. Repeating base-model training or recalibration from embeddings requires the external inputs described above.

## Rights and provenance

See `LICENSE_NOTICE.md`. The protocol files are historical experiment records; the corrected `label_free_v2` outputs and the manuscript specify the final analysis. Earlier numerical outputs are included for the paired audit and are not used to generate the final tables.
