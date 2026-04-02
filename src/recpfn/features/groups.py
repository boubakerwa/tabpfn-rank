"""Feature grouping helpers for ablations and selective training."""

from __future__ import annotations

FEATURE_SET_CHOICES = (
    "full",
    "no_user_metadata",
    "no_item_metadata",
    "no_interaction_history",
    "metadata_only",
)

USER_METADATA_FEATURES = {
    "user_age",
    "user_gender",
    "user_occupation",
}

ITEM_METADATA_FEATURES = {
    "item_primary_category",
    "item_brand",
    "item_price",
    "item_release_year",
    "item_genre_count",
    "item_category_depth",
    "price_missing",
}

INTERACTION_HISTORY_FEATURES = {
    "hist_interactions",
    "hist_positive",
    "hist_positive_rate",
    "hist_distinct_items",
    "days_since_last_interaction",
    "user_avg_history_price",
    "item_popularity",
    "item_positive_rate",
    "item_is_cold",
    "category_affinity",
    "brand_affinity",
    "same_item_history",
    "price_distance_to_user_avg",
    "history_positive_same_category",
}


def select_feature_columns(feature_cols: list[str], feature_set: str = "full") -> list[str]:
    """Select a reproducible feature subset by named feature set."""

    normalized = feature_set.strip().lower()
    if normalized not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unsupported feature set '{feature_set}'. Expected one of: {', '.join(FEATURE_SET_CHOICES)}.")

    user_features = {column for column in feature_cols if column in USER_METADATA_FEATURES}
    item_features = {
        column
        for column in feature_cols
        if column in ITEM_METADATA_FEATURES or column.startswith("feat_genre_")
    }
    interaction_features = {column for column in feature_cols if column in INTERACTION_HISTORY_FEATURES}
    metadata_features = user_features | item_features

    if normalized == "full":
        selected = set(feature_cols)
    elif normalized == "no_user_metadata":
        selected = set(feature_cols) - user_features
    elif normalized == "no_item_metadata":
        selected = set(feature_cols) - item_features
    elif normalized == "no_interaction_history":
        selected = set(feature_cols) - interaction_features
    else:
        selected = metadata_features

    ordered = [column for column in feature_cols if column in selected]
    if not ordered:
        raise ValueError(f"Feature set '{feature_set}' produced no feature columns.")
    return ordered
