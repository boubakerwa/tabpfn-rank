import math

import pandas as pd

from recpfn.eval.metrics import evaluate_rankings


def test_evaluate_rankings_matches_expected_values():
    predictions = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "item_id": [1, 2, 3, 4, 5, 6],
            "label": [1, 0, 0, 0, 1, 0],
            "score": [0.9, 0.3, 0.1, 0.8, 0.7, 0.1],
        }
    )

    metrics = evaluate_rankings(predictions)

    expected_ndcg_q1 = 1.0
    expected_ndcg_q2 = 1.0 / math.log2(3)
    assert round(metrics["ndcg@10"], 6) == round((expected_ndcg_q1 + expected_ndcg_q2) / 2, 6)
    assert metrics["recall@10"] == 1.0
    assert metrics["hitrate@10"] == 1.0
    assert round(metrics["mrr"], 6) == round((1.0 + 0.5) / 2, 6)
