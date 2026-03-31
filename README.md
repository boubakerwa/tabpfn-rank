# TabPFN-Rank

Paper-first project for testing whether TabPFN-style inference can improve candidate reranking on small-data, metadata-heavy recommendation tasks.

## Status

This repository is intentionally initialized as a research project first, not as a full product or framework.

Phase 1 exists to answer a single question:

> Is there enough real signal here to justify a deeper OSS library and an arXiv-level paper?

If the answer is no, we stop early or keep only the benchmark artifacts.

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

## Development Principle

This repo should stay brutally honest.

If the evidence is weak, the repository should say so.
If the results are strong, the project earns the right to become a cleaner public library and a stronger paper.

## Public Release Notes

- Do not bundle restricted model weights.
- Keep TabPFN as an optional backend.
- Pick a project license before the first public release.
- Keep the README aligned with the evidence, not with the ambition.
