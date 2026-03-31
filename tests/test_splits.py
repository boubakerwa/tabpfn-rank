from recpfn.data.loaders import load_dataset
from recpfn.data.splits import build_splits


def test_warm_split_has_no_future_leakage():
    dataset = load_dataset("synthetic")
    split = build_splits(dataset, "warm", seed=0)

    assert not split.test_queries.empty
    for query in split.test_queries.itertuples(index=False):
        future_train_rows = split.train_interactions[
            (split.train_interactions["user_id"] == query.user_id)
            & (split.train_interactions["timestamp"] >= query.query_timestamp)
        ]
        user_history = split.train_interactions[
            (split.train_interactions["user_id"] == query.user_id)
            & (split.train_interactions["timestamp"] < query.query_timestamp)
        ]
        assert future_train_rows.empty
        assert not user_history.empty


def test_item_cold_split_excludes_cold_items_from_training():
    dataset = load_dataset("synthetic")
    split = build_splits(dataset, "item_cold", seed=0)

    assert split.cold_item_ids
    assert set(split.train_interactions["item_id"]).isdisjoint(split.cold_item_ids)
