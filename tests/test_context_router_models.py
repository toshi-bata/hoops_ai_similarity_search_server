"""Regression tests for the context Pydantic response models.

These lock the response models against the *actual* shapes the HOOPS AI Context
Layer returns (captured from a live SDK run), so a field typed
too narrowly cannot slip through and turn into a bare 500 during response
serialization -- the endpoint builds these models after its try/except, so a
ValidationError there is not caught. No SDK or license is needed.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from routers.similarity import (  # noqa: E402
    ContextPredictHit,
    ContextPredictionEntry,
    ContextPredictResponse,
)


# The per-key coverage the SDK returns is a breakdown dict, not a scalar.
_COVERAGE = {
    "observed_hits": 8,
    "total_hits": 13,
    "observed_score_weight": 0.615055,
    "missing_score_weight": 0.384945,
}


class TestContextResponseModels(unittest.TestCase):
    def test_prediction_entry_accepts_dict_coverage(self):
        entry = ContextPredictionEntry(
            value=8.3567,
            confidence=0.7347,
            evidence_count=8,
            status="needs_review",
            coverage=_COVERAGE,
            alternatives=[{"lower": 5.88, "upper": 11.85, "std": 3.01}],
            reasons=["observed score weight below 0.67"],
            injected_context={"Process": "Cold Forming"},
        )
        self.assertEqual(entry.coverage, _COVERAGE)

    def test_full_response_serializes(self):
        preds = {
            "Material": ContextPredictionEntry(
                value=None,
                confidence=0.0,
                evidence_count=0,
                status="insufficient_evidence",
                coverage=_COVERAGE,
                alternatives=[],
                reasons=["no usable evidence"],
                injected_context=None,
            ),
            "UnitCost": ContextPredictionEntry(
                value=8.3567,
                confidence=0.7347,
                evidence_count=8,
                status="needs_review",
                coverage=_COVERAGE,
                alternatives=[{"lower": 5.88, "upper": 11.85, "std": 3.01}],
                reasons=["observed score weight below 0.67"],
                injected_context={"Process": "Cold Forming"},
            ),
        }
        hits = [
            ContextPredictHit(
                id="abc",
                score=1.0,
                filename="211.stp",
                thumbnail_url="http://x/t",
                values={"UnitCost": 9.06, "LeadTimeDays": 7},
            )
        ]
        resp = ContextPredictResponse(
            keys=["Material", "UnitCost"],
            candidates_examined=15,
            hits_considered=15,
            hits_with_context=13,
            coverage=0.866,
            predictions=preds,
            hits=hits,
        )
        # This is what FastAPI does to send the response; it must not raise.
        payload = resp.model_dump_json()
        self.assertIn("needs_review", payload)


if __name__ == "__main__":
    unittest.main()
