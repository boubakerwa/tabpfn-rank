"""Dataset loaders for supported public benchmarks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recpfn.data.schemas import (
    EVENT_COL,
    INTERACTION_ID_COL,
    ITEM_ID_COL,
    LABEL_COL,
    QUERY_INTERACTION_ID_COL,
    QUERY_TIMESTAMP_COL,
    TIMESTAMP_COL,
    USER_ID_COL,
)
from recpfn.exceptions import DatasetConfigurationError
from recpfn.types import DatasetBundle
from recpfn.utils import (
    ensure_dir,
    iter_jsonl_gz,
    maybe_download,
    maybe_download_jsonl_gz_prefix,
    read_env_int,
    unzip_file,
)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
AMAZON_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/Baby_Products.jsonl.gz"
)
AMAZON_META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_Baby_Products.jsonl.gz"
)


def load_dataset(name: str, cache_dir: str | Path = "data", seed: int = 0) -> DatasetBundle:
    """Load one of the supported datasets into the canonical schema."""

    cache_path = ensure_dir(cache_dir)
    normalized = name.lower().replace("-", "_")

    if normalized in {"movielens", "movielens_100k", "ml100k", "ml_100k"}:
        return _load_movielens_100k(cache_path)
    if normalized in {"amazon_baby", "amazon_reviews_2023_baby", "baby_products"}:
        return _load_amazon_baby(cache_path)
    if normalized in {"synthetic", "tiny"}:
        return _load_synthetic(seed=seed)

    raise DatasetConfigurationError(
        f"Unsupported dataset '{name}'. Expected one of movielens_100k, baby_products, synthetic."
    )


def _load_movielens_100k(cache_dir: Path) -> DatasetBundle:
    raw_dir = ensure_dir(cache_dir / "raw" / "movielens_100k")
    archive = maybe_download(MOVIELENS_URL, raw_dir / "ml-100k.zip")
    extracted = unzip_file(archive, raw_dir / "extracted")
    base_dir = extracted / "ml-100k"

    users = pd.read_csv(
        base_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )

    genre_cols = [
        "genre_unknown",
        "genre_action",
        "genre_adventure",
        "genre_animation",
        "genre_children",
        "genre_comedy",
        "genre_crime",
        "genre_documentary",
        "genre_drama",
        "genre_fantasy",
        "genre_film_noir",
        "genre_horror",
        "genre_musical",
        "genre_mystery",
        "genre_romance",
        "genre_sci_fi",
        "genre_thriller",
        "genre_war",
        "genre_western",
    ]
    items = pd.read_csv(
        base_dir / "u.item",
        sep="|",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            *genre_cols,
        ],
        encoding="latin-1",
    )
    items["release_date"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce")
    items["release_year"] = items["release_date"].dt.year.fillna(0).astype(int)
    items["primary_category"] = items[genre_cols].idxmax(axis=1).str.replace("genre_", "", regex=False)
    items["genre_count"] = items[genre_cols].sum(axis=1)

    interactions = pd.read_csv(
        base_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    interactions[TIMESTAMP_COL] = pd.to_datetime(interactions[TIMESTAMP_COL], unit="s", utc=True)
    interactions[INTERACTION_ID_COL] = range(len(interactions))
    interactions[EVENT_COL] = "rating"
    interactions[LABEL_COL] = (interactions["rating"] >= 4).astype(int)

    return DatasetBundle(
        name="movielens_100k",
        users=users,
        items=items,
        interactions=interactions,
        user_feature_columns=["age", "gender", "occupation"],
        item_feature_columns=["primary_category", "release_year", "genre_count", *genre_cols],
        metadata={"raw_dir": str(base_dir), "positive_rule": "rating >= 4"},
    )


def _load_amazon_baby(cache_dir: Path) -> DatasetBundle:
    raw_dir = ensure_dir(cache_dir / "raw" / "amazon_reviews_2023_baby")
    max_reviews = read_env_int("RECPFN_AMAZON_MAX_REVIEWS")
    max_meta = read_env_int("RECPFN_AMAZON_MAX_META")

    if max_reviews is not None:
        reviews_path = maybe_download_jsonl_gz_prefix(
            AMAZON_REVIEW_URL,
            raw_dir / f"Baby_Products.head_{max_reviews}.jsonl.gz",
            max_lines=max_reviews,
        )
    else:
        reviews_path = maybe_download(AMAZON_REVIEW_URL, raw_dir / "Baby_Products.jsonl.gz")

    if max_meta is not None:
        meta_path = maybe_download_jsonl_gz_prefix(
            AMAZON_META_URL,
            raw_dir / f"meta_Baby_Products.head_{max_meta}.jsonl.gz",
            max_lines=max_meta,
        )
    else:
        meta_path = maybe_download(AMAZON_META_URL, raw_dir / "meta_Baby_Products.jsonl.gz")

    reviews = []
    for idx, row in enumerate(iter_jsonl_gz(reviews_path)):
        timestamp_ms = row.get("timestamp") or row.get("timestamp_ms")
        reviews.append(
            {
                INTERACTION_ID_COL: idx,
                USER_ID_COL: row.get("user_id") or row.get("reviewerID"),
                ITEM_ID_COL: row.get("parent_asin") or row.get("asin"),
                "rating": row.get("rating") or row.get("overall"),
                TIMESTAMP_COL: pd.to_datetime(timestamp_ms, unit="ms", utc=True),
                EVENT_COL: "review",
                LABEL_COL: int((row.get("rating") or row.get("overall") or 0) >= 4),
            }
        )
    interactions = pd.DataFrame(reviews).dropna(subset=[USER_ID_COL, ITEM_ID_COL, TIMESTAMP_COL])

    items = []
    review_item_ids = set(interactions[ITEM_ID_COL].unique().tolist()) if not interactions.empty else set()
    for idx, row in enumerate(iter_jsonl_gz(meta_path)):
        item_id = row.get("parent_asin") or row.get("asin")
        if review_item_ids and item_id not in review_item_ids:
            continue
        categories = row.get("categories") or []
        flat_categories = [part for cat in categories for part in cat] if categories and isinstance(categories[0], list) else categories
        primary_category = flat_categories[-1] if flat_categories else "unknown"
        price = row.get("price")
        if isinstance(price, str):
            cleaned = price.replace("$", "").replace(",", "").strip()
            try:
                price = float(cleaned)
            except ValueError:
                price = None
        items.append(
            {
                ITEM_ID_COL: item_id,
                "title": row.get("title"),
                "brand": row.get("store") or row.get("brand") or "unknown",
                "price": price,
                "primary_category": primary_category,
                "category_depth": len(flat_categories),
            }
        )
    items = pd.DataFrame(items).drop_duplicates(subset=[ITEM_ID_COL])
    items["price"] = pd.to_numeric(items["price"], errors="coerce")
    items["brand"] = items["brand"].fillna("unknown").astype(str)
    items["primary_category"] = items["primary_category"].fillna("unknown").astype(str)

    users = pd.DataFrame({USER_ID_COL: interactions[USER_ID_COL].drop_duplicates().sort_values()})

    return DatasetBundle(
        name="amazon_baby_products",
        users=users,
        items=items,
        interactions=interactions,
        user_feature_columns=[],
        item_feature_columns=["brand", "price", "primary_category", "category_depth"],
        metadata={
            "raw_dir": str(raw_dir),
            "positive_rule": "rating >= 4",
            "max_reviews": max_reviews,
            "max_meta": max_meta,
        },
    )


def _load_synthetic(seed: int = 0) -> DatasetBundle:
    del seed
    users = pd.DataFrame(
        {
            USER_ID_COL: [1, 2, 3],
            "age": [24, 31, 45],
            "gender": ["F", "M", "F"],
            "occupation": ["artist", "engineer", "teacher"],
        }
    )
    items = pd.DataFrame(
        {
            ITEM_ID_COL: [101, 102, 103, 104, 105, 106, 107, 108],
            "primary_category": ["action", "action", "drama", "drama", "family", "family", "action", "drama"],
            "price": [10.0, 12.0, 8.0, 9.0, 7.0, 11.0, 13.0, 6.0],
            "brand": ["a", "a", "b", "b", "c", "c", "a", "b"],
        }
    )
    interactions = pd.DataFrame(
        {
            INTERACTION_ID_COL: range(12),
            USER_ID_COL: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            ITEM_ID_COL: [101, 103, 102, 105, 102, 104, 106, 101, 103, 104, 105, 106],
            "rating": [5, 4, 2, 5, 5, 4, 1, 4, 5, 2, 4, 5],
            TIMESTAMP_COL: pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC"),
            EVENT_COL: "review",
        }
    )
    interactions[LABEL_COL] = (interactions["rating"] >= 4).astype(int)

    return DatasetBundle(
        name="synthetic",
        users=users,
        items=items,
        interactions=interactions,
        user_feature_columns=["age", "gender", "occupation"],
        item_feature_columns=["primary_category", "price", "brand"],
        metadata={"positive_rule": "rating >= 4"},
    )
