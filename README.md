<h1 align="center">TabPFN-Rank</h1>

<p align="center">
  <strong>Small-data candidate reranking with TabPFN and strong tabular baselines.</strong>
</p>

<p align="center">
  Paper-first benchmark and method project for testing whether TabPFN-style inference can improve metadata-heavy recommendation reranking.
</p>

<p align="center">
  <a href="https://github.com/boubakerwa/tabpfn-rank"><img alt="repo" src="https://img.shields.io/badge/repo-tabpfn--rank-111827"></a>
  <img alt="phase" src="https://img.shields.io/badge/phase-2%20pointwise%20validated-0f766e">
  <img alt="decision" src="https://img.shields.io/badge/decision-pointwise%20story%20holds-2563eb">
  <img alt="focus" src="https://img.shields.io/badge/focus-targeted%20native%20adapter-7c3aed">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-0ea5e9">
  <img alt="datasets" src="https://img.shields.io/badge/datasets-MovieLens%20%7C%20Amazon-f59e0b">
</p>

<p align="center">
  <a href="#research-snapshot">Snapshot</a> •
  <a href="#phase-2-findings">Findings</a> •
  <a href="#tracked-results">Tracked Results</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#repo-layout">Repo Layout</a> •
  <a href="#the-tipping-point">Decision Rules</a>
</p>

## Research Snapshot

> **Current call:** drill further, but center the story on **pointwise TabPFN** and treat pairwise TabPFN as an ablation unless later evidence changes that.

| Signal | Current read |
| --- | --- |
| Main evidence state | Phase 2 pointwise sweep complete and summarized |
| Strongest method story | Pointwise TabPFN for small-data and cold-start reranking |
| Strongest benchmark regime | `MovieLens 100K` warm/global and item-cold |
| Native adapter outcome | `tabpfn_native` is a targeted cold-start/global-popularity variant |
| Tracked artifact bundle | `paper/phase2_pointwise/` |
| Main caution | Trees still dominate runtime, and `tabpfn_native` is not a universal replacement |

## Status

This repository is still intentionally run as a research project first, not as a polished product library.

Phase 1 is frozen. The Phase 2 pointwise validation sweep is now complete, the reporting layer is restored, and the current state of the evidence is tracked in-repo.

## Phase 2 Findings

- On the four key `MovieLens 100K` pointwise slices at `100%` train, `K=20`, and `3` seeds, a TabPFN variant beat the best tree baseline every time:
  - `warm / global_popularity`: `tabpfn_native 0.8416` vs `catboost 0.7956`
  - `warm / context_popularity`: `tabpfn 0.5431` vs `catboost 0.5023`
  - `item_cold / global_popularity`: `tabpfn_native 0.9852` vs `catboost 0.9740`
  - `item_cold / context_popularity`: `tabpfn 0.8286` vs `catboost 0.7741`
- The overall pointwise TabPFN story holds. The best TabPFN variant stayed within `1%` of the best tree on `4/4` key MovieLens slices, and in practice beat it on all four.
- `tabpfn_native` is useful but not universal. It had positive mean `NDCG@10` deltas on `2/4` key slices and a `0.782x` median runtime ratio versus the one-hot TabPFN path, so it is best treated as a targeted cold-start / global-popularity adapter rather than a full replacement.
- The low-data story is still real. On `MovieLens warm / global_popularity`, a TabPFN variant was best at `10%`, `20%`, `50%`, and `100%`. On `MovieLens item_cold / global_popularity`, `tabpfn_native` beat the one-hot path at every train fraction from `10%` to `100%`.
- Candidate-pool sensitivity is encouraging but not uniformly easy. On `item_cold / global_popularity`, `tabpfn_native` retained `1.006x` of its `K=20` quality at `K=50` and still reached `0.9753` at `K=100`. Warm and context-heavy slices degrade more as `K` grows.
- Amazon remains secondary evidence. The `context_popularity` slices support the same direction:
  - `warm`: `tabpfn_native 0.5697`
  - `item_cold`: `tabpfn_native 0.9380`
  while `global_popularity` is saturated and should not drive the main claim.
- Feature-group ablations suggest the gains are not just coming from user demographics. On `MovieLens item_cold`, removing interaction history or using metadata-only causes a large drop for both TabPFN variants, while removing user metadata changes little.

## Current Read

- The project is worth pushing forward.
- The main story is **pointwise TabPFN for small-data and cold-start reranking**.
- `tabpfn_native` should be described as a **targeted adapter variant**, not the new default.
- Pairwise TabPFN stays out of the headline unless later evidence changes that.

## Core Research Question

Can TabPFN, used as a pointwise or pairwise reranker, beat strong tabular baselines on:

- small-data recommendation
- item cold-start
- user cold-start
- metadata-heavy reranking

The practical framing is narrow on purpose:

- retrieval is out of scope
- candidate reranking is in scope
- public datasets only for Phase 1
- low-friction, reproducible Python experiments only

## What This Project Is

- a Python package for research code and later reusable reranking components
- a benchmark harness for public small-data recommendation datasets
- a paper workspace for results tables, figures, and a short technical note

## What This Project Is Not

- not a general-purpose recommender platform
- not a retrieval engine
- not a large-scale production recsys stack
- not a new foundation model training project
- not a months-long build before first evidence

## Phase Plan

### Phase 1: Validate Or Kill

Completed. This phase established the benchmark spine, the dataset loaders, the split builders, and enough early evidence to justify a deeper pointwise validation pass.

### Phase 2: Pointwise Validation

Completed. This phase added the multi-seed pointwise matrix, bootstrap deltas, `K`-sensitivity, Amazon sanity checks, and feature-group ablations that now back the current read.

### Next: Decision + Writeup

The next project step is to turn the Phase 2 evidence bundle into a narrow decision about scope:

1. keep the paper centered on **pointwise TabPFN**,
2. present `tabpfn_native` as a targeted adapter result,
3. decide how much additional method work is actually justified before drafting.

## The Tipping Point

Phase 1 has already crossed this threshold. The current Phase 2 pointwise decision is tracked in `paper/phase2_pointwise/decision.md`.

We continue only if at least one of the following is true:

1. Pairwise TabPFN beats pointwise TabPFN by a clearly meaningful margin on at least one key regime.
   Practical threshold: about `+0.02` absolute `NDCG@10`, or a similarly meaningful gain in `Recall@10`, on warm-start, item cold-start, or a low-data setting.

2. A TabPFN variant is best, or close enough to best to be genuinely interesting, on at least one hard setting.
   Practical threshold: best result or within roughly `1%` relative of the best baseline on an item cold-start or low-data benchmark, with runtime still usable at small candidate-set sizes.

3. The benchmark produces a sharp, defensible finding that is worth writing up even if it is partly negative.
   Example: TabPFN only becomes competitive in a narrow low-data or metadata-rich regime that standard baselines miss.

If none of those happen, we do not keep expanding the method story.

## Stop Conditions

We stop deeper method development and keep this as a small benchmark repo only if all of the following are true:

- pairwise TabPFN does not materially beat pointwise TabPFN anywhere
- XGBoost and CatBoost dominate both quality and runtime across the tested settings
- any apparent gains disappear once candidate-set construction is held constant

That outcome is still useful, but it does not justify a bigger library or a stronger paper claim.

## Initial Evaluation Rules

To avoid fooling ourselves:

- keep retrieval out of scope in the MVP
- fix candidate-set size across models in the main benchmark
- always include the ground-truth positive in evaluation candidate sets
- use at least one popularity-based baseline that is hard to beat
- report runtime, because TabPFN can lose badly there
- treat item cold-start as especially important
- treat user cold-start as optional if the dataset metadata is too weak

## Repo Layout

```text
src/recpfn/       Research package and future reusable components
experiments/      Reproducible experiment entry points
configs/          Config files and experiment settings
paper/            Figures, result tables, and manuscript assets
tests/            Sanity checks and regression tests
```

## Current Implementation

The repo now includes the MVP2 first-sweep benchmark spine:

- canonical dataset loaders for `MovieLens 100K`, `Amazon Reviews 2023 / Baby_Products`, and a tiny synthetic test dataset
- deterministic `warm` and `item_cold` split builders
- `global_popularity` and `context_popularity` candidate protocols with oracle positive inclusion
- compact tabular feature builder with user history, item metadata, and user-item affinity features
- pointwise and pairwise training flows
- optional model adapters for `XGBoost`, `CatBoost`, and `TabPFN`
- two TabPFN adapter paths for comparison: the existing one-hot path (`tabpfn`) and an experimental native-categorical path (`tabpfn_native`)
- benchmark reporting to per-query CSVs, per-run JSON metrics, summary CSV, and markdown benchmark tables
- a dedicated Phase 2 pointwise workflow:
  - `recpfn.phase2_pointwise_run` for raw multi-seed benchmark units
  - `recpfn.phase2_pointwise_report` for aggregation, bootstrap delta summaries, plots, and the Phase 2 decision memo

## Tracked Results

The main Phase 2 evidence bundle is now tracked in-repo:

- `paper/phase2_pointwise/decision.md`
- `paper/phase2_pointwise/benchmark.md`
- `paper/phase2_pointwise/aggregated_results.csv`
- `paper/phase2_pointwise/bootstrap_delta_summary.csv`
- `paper/phase2_pointwise/k_sensitivity_results.csv`
- `paper/phase2_pointwise/amazon_sanity_results.csv`
- `paper/phase2_pointwise/feature_group_ablation.csv`
- `paper/phase2_pointwise/raw_summary.csv`
- `paper/phase2_pointwise/best_tree_selection.json`
- `paper/phase2_pointwise/raw_results_archive_manifest.json`
- `paper/figures/phase2_pointwise/adapter_delta_by_train_fraction.png`
- `paper/figures/phase2_pointwise/runtime_by_train_fraction.png`
- `paper/figures/phase2_pointwise/metric_by_k.png`
- `paper/figures/phase2_pointwise/native_minus_one_hot_by_slice.png`
- `paper/figures/phase2_pointwise/best_tabpfn_vs_best_tree.png`

The raw Phase 2 per-unit run tree under `paper/results_phase2_pointwise_runs/` is intentionally left untracked because it is large (`~421MB`) and reproducible from the committed code. Major milestone raw trees are preserved as compressed GitHub Release assets, with provenance tracked in `paper/phase2_pointwise/raw_results_archive_manifest.json`.

## Quickstart

Create a Python 3.10+ environment, then install the package:

```bash
pip install -e ".[dev]"
pip install xgboost catboost
```

Run a first MovieLens benchmark sweep:

```bash
python -m recpfn.cli \
  --dataset movielens_100k \
  --split warm \
  --protocols global_popularity context_popularity \
  --pointwise-models popularity xgboost catboost \
  --max-train-queries 200 \
  --max-test-queries 100
```

Run the full Phase 2 pointwise sweep and reporting flow:

```bash
RECPFN_TABPFN_VERSION=v2.5 TABPFN_ALLOW_CPU_LARGE_DATASET=1 \
python -m recpfn.phase2_pointwise_run \
  --run-output-dir paper/results_phase2_pointwise_runs

python -m recpfn.phase2_pointwise_report \
  --run-output-dir paper/results_phase2_pointwise_runs \
  --output-dir paper/phase2_pointwise \
  --plots-output-dir paper/figures/phase2_pointwise
```

Raw Phase 2 units are written under `paper/results_phase2_pointwise_runs/`, and the tracked summaries land in `paper/phase2_pointwise/`.

To summarize an already completed Phase 1 sweep without rerunning it:

```bash
python -m recpfn.phase1_decision \
  --run-output-dir paper/results_phase1_decision_runs_final \
  --output-dir paper/phase1_decision \
  --reuse-existing
```

This Phase 2 workflow keeps Phase 1 frozen and focuses on:

- multi-seed `MovieLens 100K` pointwise validation
- `tabpfn` vs `tabpfn_native` head-to-head comparison
- paired query-level bootstrap deltas on key slices
- targeted `K=20/50/100` sensitivity on MovieLens
- a capped Amazon pointwise sanity pass
- feature-group ablations on the best cold-start slices

For the Amazon loader, you can cap raw ingestion during development:

```bash
export RECPFN_AMAZON_MAX_REVIEWS=50000
export RECPFN_AMAZON_MAX_META=100000
```

## Development Principle

This repo should stay brutally honest.

If the evidence is weak, the repository should say so.
If the results are strong, the project earns the right to become a cleaner public library and a stronger paper.

## Public Release Notes

- Do not bundle restricted model weights.
- Keep TabPFN as an optional backend.
- Pick a project license before the first public release.
- Keep the README aligned with the evidence, not with the ambition.
