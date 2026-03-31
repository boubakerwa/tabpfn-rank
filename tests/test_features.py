from recpfn.data.loaders import load_dataset
from recpfn.data.splits import build_splits
from recpfn.features.builders import build_features
from recpfn.rerank.candidate_sets import build_candidates


def test_feature_builder_emits_core_columns():
    dataset = load_dataset("synthetic")
    split = build_splits(dataset, "warm", seed=0)
    candidates = build_candidates(dataset, split, protocol="global_popularity", k=4, seed=0)
    features = build_features(dataset, candidates, split)

    expected = {
        "hist_interactions",
        "item_popularity",
        "category_affinity",
        "price_distance_to_user_avg",
        "item_primary_category",
    }
    assert expected.issubset(features.columns)
    assert features["label"].isin([0, 1]).all()
