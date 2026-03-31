"""Cold-start feature helpers."""

from __future__ import annotations

import pandas as pd

from recpfn.data.schemas import ITEM_ID_COL, LABEL_COL


def item_training_popularity(train_interactions: pd.DataFrame) -> pd.Series:
    """Training-set positive popularity per item."""

    return (
        train_interactions[train_interactions[LABEL_COL] == 1]
        .groupby(ITEM_ID_COL)
        .size()
        .rename("item_popularity")
    )
