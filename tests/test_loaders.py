from recpfn.data.loaders import load_dataset


def test_load_dataset_accepts_canonical_amazon_baby_products_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("RECPFN_AMAZON_MAX_REVIEWS", "5")
    monkeypatch.setenv("RECPFN_AMAZON_MAX_META", "5")

    dataset = load_dataset("amazon_baby_products", cache_dir=tmp_path, seed=0)

    assert dataset.name == "amazon_baby_products"
    assert not dataset.interactions.empty
