"""Infinite evidence must not disappear from otherwise flagged synthetic records."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fcontracts.contracts import FeatureContract, top3_from_evidence


class InfiniteEvidence(unittest.TestCase):
    def test_positive_infinity_is_the_strongest_violated_evidence(self):
        ev = np.array([[2.0, np.inf, 4.0, 8.0]])
        viol = np.ones(ev.shape, bool)
        np.testing.assert_array_equal(top3_from_evidence(ev, viol), [[1, 3, 2]])

    def test_unviolated_infinity_remains_excluded(self):
        ev = np.array([[np.inf, 3.0, 2.0]])
        viol = np.array([[False, True, False]])
        np.testing.assert_array_equal(top3_from_evidence(ev, viol), [[1, -1, -1]])

    def test_one_and_two_feature_rows_keep_infinite_evidence_then_padding(self):
        np.testing.assert_array_equal(
            top3_from_evidence(np.array([[np.inf]]), np.array([[True]])), [[0, -1, -1]])
        np.testing.assert_array_equal(
            top3_from_evidence(np.array([[np.inf, 2.0]]), np.array([[True, True]])), [[0, 1, -1]])

    def test_multiple_infinities_keep_valid_indices_without_requiring_tie_order(self):
        ev = np.array([[np.inf, 4.0, np.inf, 2.0]])
        result = top3_from_evidence(ev, np.ones(ev.shape, bool))[0]
        self.assertEqual(set(result[:2]), {0, 2})
        self.assertEqual(result[2], 1)

    def test_clear_and_unavailable_evidence_still_uses_minus_one(self):
        ev = np.array([[np.inf, 2.0], [-np.inf, np.nan]])
        viol = np.array([[False, False], [True, True]])
        np.testing.assert_array_equal(top3_from_evidence(ev, viol), [[-1] * 3, [-1] * 3])

    def test_continuous_infinite_input_keeps_flag_hard_violation_and_feature(self):
        rng = np.random.default_rng(32)
        reference = rng.normal(size=(800, 2))
        contract = FeatureContract([], alpha=0.05, relations=False).fit(
            reference[:400], reference[400:])
        stream = np.array([[np.inf, 0.0], [-np.inf, 0.0], [0.0, np.inf]])
        before = stream.copy()
        for chunk in (1, 2, 20):
            result = contract.check(stream, chunk=chunk)
            np.testing.assert_array_equal(result.flag, [True, True, True])
            np.testing.assert_array_equal(result.top, [[0, -1, -1], [0, -1, -1], [1, -1, -1]])
            np.testing.assert_array_equal(result.hard_rows, [0, 1, 2])
            np.testing.assert_array_equal(result.hard_cols, [0, 0, 1])
        np.testing.assert_array_equal(stream, before)


if __name__ == "__main__":
    unittest.main()
