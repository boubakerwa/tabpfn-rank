from recpfn.data.loaders import load_dataset
from recpfn.data.splits import build_splits
from recpfn.rerank.candidate_sets import build_candidates


def test_candidates_force_include_positive_and_fixed_k():
    dataset = load_dataset("synthetic")
    split = build_splits(dataset, "warm", seed=0)
    candidates = build_candidates(dataset, split, protocol="global_popularity", k=4, seed=0)

    grouped = candidates.groupby(["query_id", "split"])
    for (_, _), group in grouped:
        assert len(group) == 4
        assert group["label"].sum() == 1


def test_context_protocol_builds_rows():
    dataset = load_dataset("synthetic")
    split = build_splits(dataset, "warm", seed=0)
    candidates = build_candidates(dataset, split, protocol="context_popularity", k=4, seed=0)

    assert not candidates.empty
    assert set(candidates["protocol"]) == {"context_popularity"}
