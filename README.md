# TabPFN-Rank

Paper-first project for testing whether TabPFN-style inference can improve candidate reranking on small-data, metadata-heavy recommendation tasks.

## Status

This repository is intentionally initialized as a research project first, not as a full product or framework.

Phase 1 exists to answer a single question:

> Is there enough real signal here to justify a deeper OSS library and an arXiv-level paper?

If the answer is no, we stop early or keep only the benchmark artifacts.

## Phase 1 Findings

Phase 1 now has a clear decision: **drill further**, but center the next phase on **pointwise TabPFN**, not pairwise TabPFN.

- The strongest positive signal is the low-data warm-start ladder on `MovieLens`. Pointwise TabPFN was best at `10%`, `20%`, and `50%` train scale with `NDCG@10 = 0.8362`, `0.8306`, and `0.8315`.
- The strongest cold-start pointwise signal remains `MovieLens item_cold / context_popularity`, where pointwise TabPFN reached `0.9087` vs the best tree baseline at `0.8755`.
- The Amazon context-aware signal is still positive but secondary: `Amazon warm / context_popularity` pointwise TabPFN reached `0.6143` vs the best tree at `0.4646`.
- Pairwise TabPFN is not the lead story. In the low-data ladder it only became best at `MovieLens warm / 100%` (`0.8412`), and it stayed expensive throughout.
- On `MovieLens item_cold / global_popularity`, pairwise TabPFN was strong at full data (`0.9742`) but still slightly behind the best tree (`0.9769`).
- The missing canonical block remains the narrow `amazon_baby_products / item_cold / context_popularity / pairwise` corner, and it is non-blocking for the Phase 1 decision.

Important caveats:

- In the low-data ladder, pairwise TabPFN runtimes ranged from about `141s` to `407s`, versus roughly `6s` to `24s` for pointwise TabPFN.
- `Amazon / global_popularity` is partly saturated and should not drive the main paper claim.
- The current Amazon result is still provisional because the local run used a capped review slice.

## Adapter Comparison

We also added a parallel experimental adapter, `tabpfn_native`, that keeps categorical columns in mixed-type tabular form instead of routing TabPFN through the shared one-hot path.

- On `MovieLens 100K` pointwise reranking, `tabpfn_native` was strongest on the `item_cold / global_popularity` ladder: it beat the existing one-hot TabPFN path at `10%`, `20%`, `50%`, and `100%` train scale, with `NDCG@10` gains of `+0.0417`, `+0.0314`, `+0.0044`, and `+0.0235`.
- It was also faster in every one of those item-cold runs.
- The effect is regime-dependent, not universal: the native path lost on the `MovieLens` `context_popularity` pointwise checks and on the initial pairwise comparison.
- A small capped Amazon pointwise check (`25/25` queries) was also mixed: `tabpfn_native` won on `warm / context_popularity` (`0.7507` vs `0.7102`) and was dramatically faster, but the original one-hot path won on `item_cold / context_popularity` (`0.7903` vs `0.7045`).

Current interpretation: **TabPFN feature representation is itself an important experimental variable**, and native categorical handling looks most promising for **pointwise cold-start reranking**, not for pairwise ranking.

## Immediate Next Step

The next step is no longer more Phase 1 benchmarking. It is to freeze Phase 1 and start MVP 3 with the right scope:

1. Treat **pointwise TabPFN for small-data and cold-start reranking** as the main method story.
2. Keep pairwise TabPFN as an ablation unless later targeted evidence changes that.
3. Turn the current benchmark table, memo, and figure set into one concise writeup.

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

Goal: get enough evidence, quickly, to decide whether this deserves deeper investment.

Minimum Phase 1 deliverables:

1. Two public datasets with reproducible loaders.
2. Fixed candidate-set generation protocol for each dataset.
3. Warm-start, item cold-start, and if feasible user cold-start splits.
4. Baselines:
   - popularity / recent popularity
   - XGBoost pointwise
   - CatBoost pointwise
   - XGBoost ranking or pairwise baseline
   - CatBoost ranking or pairwise baseline
   - TabPFN pointwise
   - TabPFN pairwise
5. Metrics:
   - NDCG@10
   - Recall@10
   - MRR
   - HitRate@10
   - runtime / latency
6. One compact benchmark table that we would be comfortable showing publicly.

Planned dataset path:

- `MovieLens 100K`
- one metadata-rich public dataset from `Amazon Reviews 2023`

Fallback if Amazon setup becomes awkward for the MVP:

- `MIND-small` as a reranking-style benchmark with impression groups already defined

### Phase 2: Drill Further Only If Earned

We only move into deeper method and library work if Phase 1 produces a credible signal.

Possible Phase 2 work:

- cleaner public Python API
- better experiment CLI
- pairwise TabPFN scoring refinements
- list-context ablations
- stronger cold-start protocols
- tighter writeup targeting an arXiv preprint

## The Tipping Point

We continue past Phase 1 only if at least one of the following is true:

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

Outputs are written under `paper/results/<dataset>/<split>/`.

To summarize an already completed Phase 1 sweep without rerunning it:

```bash
python -m recpfn.phase1_decision \
  --run-output-dir paper/results_phase1_decision_runs_final \
  --output-dir paper/phase1_decision \
  --reuse-existing
```

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
