"""Unit tests for the core library: calibration helpers, corruption operators, paging and
the online contract check. All data is synthetic and seeded; no datasets are needed.

    python3 -m unittest discover -s tests -v

Needs numpy, scipy and scikit-learn (see requirements.txt).
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fcontracts.config import CONFIG                                   # noqa: E402
from fcontracts.contracts import (FeatureContract, RateCheck,           # noqa: E402
                                  conformal_threshold, top3_from_evidence,
                                  wilson_upper)
from fcontracts.inject import inject, reference_stats                   # noqa: E402
from fcontracts.paging import (burst_aware_pages, onset_stats,          # noqa: E402
                               page_times_one, p1_for, run_paging)


def synthetic(n, rng):
    a = rng.lognormal(3.0, 0.6, n)
    b = 2.0 * a + rng.normal(0, 1.5, n)
    c = rng.integers(0, 4, n).astype(float)
    e = rng.normal(0, 1, n)
    return np.column_stack([a, b, c, e])


def fitted(calibration="conformal", seed=0):
    rng = np.random.default_rng(seed)
    ref = synthetic(25000, rng)
    c = FeatureContract(groups=[[0, 1]], alpha=1e-3, calibration=calibration)
    return c.fit(ref[:20000], ref[20000:]), ref, rng


class CalibrationHelpers(unittest.TestCase):
    def test_wilson_upper_bounds_the_rate(self):
        for k, n in [(0, 100), (1, 100), (10, 100), (99, 100), (100, 100)]:
            u = wilson_upper(k, n)
            self.assertGreaterEqual(u + 1e-12, k / n)
            self.assertLessEqual(u, 1.0)

    def test_wilson_upper_empty_sample_is_one(self):
        self.assertEqual(wilson_upper(0, 0), 1.0)

    def test_wilson_upper_shrinks_with_more_data(self):
        self.assertLess(wilson_upper(0, 10000), wilson_upper(0, 100))

    def test_conformal_threshold_picks_order_statistic(self):
        scores = np.arange(1, 100, dtype=float)             # n = 99
        # ceil(100 * 0.9) = 90th smallest
        self.assertEqual(conformal_threshold(scores, 0.1), 90.0)

    def test_conformal_threshold_is_infinite_when_sample_too_small(self):
        self.assertEqual(conformal_threshold([1.0, 2.0, 3.0], 0.01), math.inf)

    def test_conformal_threshold_ignores_input_order(self):
        rng = np.random.default_rng(1)
        s = rng.normal(size=500)
        self.assertEqual(conformal_threshold(s, 0.05), conformal_threshold(s[::-1], 0.05))

    def test_top3_returns_highest_evidence_violations_first(self):
        ev = np.array([[0.1, 5.0, 3.0, 4.0, 1.0]])
        viol = np.array([[True, True, True, True, False]])
        self.assertEqual(top3_from_evidence(ev, viol)[0].tolist(), [1, 3, 2])

    def test_top3_pads_with_minus_one_when_few_violations(self):
        ev = np.ones((1, 4))
        viol = np.array([[False, True, False, False]])
        self.assertEqual(top3_from_evidence(ev, viol)[0].tolist(), [1, -1, -1])

    def test_top3_all_clear_row(self):
        out = top3_from_evidence(np.ones((2, 5)), np.zeros((2, 5), bool))
        self.assertTrue((out == -1).all())


class CorruptionOperators(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.X = synthetic(2000, self.rng)
        self.ref = reference_stats(self.X)

    def run_fault(self, fault, prevalence=0.1, features=(1,), **kw):
        return inject(self.X, fault, list(features), prevalence,
                      np.random.default_rng(7), self.ref, **kw)

    def test_input_is_not_modified(self):
        before = self.X.copy()
        self.run_fault("scale")
        np.testing.assert_array_equal(before, self.X)

    def test_only_selected_feature_changes(self):
        X2, changed = self.run_fault("scale", features=(1,))
        self.assertFalse(changed[:, [0, 2, 3]].any())
        self.assertTrue(changed[:, 1].any())

    def test_prevalence_controls_number_of_rows(self):
        _, changed = self.run_fault("scale", prevalence=0.1)
        self.assertEqual(int(changed[:, 1].sum()), 200)

    def test_scale_multiplies_by_factor(self):
        X2, changed = self.run_fault("scale", scale=10.0)
        np.testing.assert_allclose(X2[changed[:, 1], 1], self.X[changed[:, 1], 1] * 10.0)

    def test_nan_fill_sets_missing(self):
        X2, changed = self.run_fault("nan_fill")
        self.assertTrue(np.isnan(X2[changed[:, 1], 1]).all())

    def test_default_fill_sets_zero(self):
        X2, changed = self.run_fault("default_fill")
        self.assertTrue((X2[changed[:, 1], 1] == 0).all())

    def test_magnitude_is_far_outside_range(self):
        X2, changed = self.run_fault("magnitude")
        self.assertGreater(X2[changed[:, 1], 1].min(), 1e7)

    def test_clipping_never_exceeds_quantile(self):
        X2, _ = self.run_fault("clipping", prevalence=1.0)
        self.assertLessEqual(X2[:, 1].max(), self.ref["q90"][1] + 1e-9)

    def test_shift_stays_inside_reference_range(self):
        X2, _ = self.run_fault("shift", prevalence=1.0)
        self.assertGreaterEqual(X2[:, 1].min(), self.ref["lo"][1])
        self.assertLessEqual(X2[:, 1].max(), self.ref["hi"][1])

    def test_misjoin_preserves_the_set_of_values(self):
        X2, _ = self.run_fault("misjoin", prevalence=1.0)
        np.testing.assert_allclose(np.sort(X2[:, 1]), np.sort(self.X[:, 1]))

    def test_encoding_flips_sign_for_continuous_features(self):
        X2, changed = self.run_fault("encoding", features=(1,))
        np.testing.assert_allclose(X2[changed[:, 1], 1], -self.X[changed[:, 1], 1])

    def test_unknown_fault_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fault("not_a_fault")

    def test_same_seed_gives_same_corruption(self):
        a, _ = self.run_fault("inconsistency")
        b, _ = self.run_fault("inconsistency")
        np.testing.assert_array_equal(a, b)


class PagingTests(unittest.TestCase):
    P0, THRESH = 0.001, 1e4

    def pages(self, events, test="sr"):
        p0 = self.P0
        return page_times_one(np.asarray(events), p0, float(p1_for(p0)), self.THRESH, test)

    def test_no_events_no_pages(self):
        self.assertEqual(self.pages([]), [])

    def test_isolated_violations_do_not_page(self):
        self.assertEqual(self.pages([100, 5000, 12000]), [])

    def test_dense_run_pages_and_restarts(self):
        pages = self.pages(list(range(1000, 1300)))
        self.assertGreaterEqual(len(pages), 1)
        self.assertTrue(all(1000 <= t < 1300 for t in pages))
        self.assertEqual(pages, sorted(pages))

    def test_cusum_also_pages_on_dense_run(self):
        self.assertGreaterEqual(len(self.pages(list(range(1000, 1300)), "cusum")), 1)

    def test_sr_pages_sooner_with_denser_violations(self):
        dense = self.pages(list(range(1000, 1500)))[0]
        sparse = self.pages(list(range(1000, 3000, 2)))[0] - 1000
        self.assertLess(dense - 1000, sparse)

    def test_run_paging_reports_the_feature(self):
        rows = np.arange(500, 800)
        cols = np.full(rows.size, 2)
        out = run_paging(rows, cols, 4, np.full(4, self.P0), A=1e3, test="sr")
        self.assertTrue(out)
        self.assertTrue(all(j == 2 for _, j in out))

    def test_run_paging_is_sorted_by_time(self):
        rows = np.r_[np.arange(500, 800), np.arange(100, 400)]
        cols = np.r_[np.zeros(300, int), np.ones(300, int)]
        out = run_paging(rows, cols, 3, np.full(3, self.P0), A=1e3, test="sr")
        self.assertEqual(out, sorted(out))

    def test_onset_stats_counts_runs(self):
        rows = np.array([10, 11, 12, 50, 90, 91])
        cols = np.zeros(6, int)
        _, max_run = onset_stats(rows, cols, 2, 1000)
        self.assertEqual(max_run[0], 3)
        self.assertEqual(max_run[1], 0)

    def test_burst_aware_pages_once_per_long_run(self):
        rows = np.arange(1000, 1100)               # one 100-long run on feature 0
        cols = np.zeros(100, int)
        pages = burst_aware_pages(rows, cols, 2, np.array([1e-3, 1e-3]), np.array([2.0, 2.0]),
                                  A=1e12, min_run=20)
        run_pages = [t for t, j in pages if j == 0]
        self.assertEqual(run_pages, [1020])        # run-length rule only, SR threshold unreachable

    def test_burst_aware_short_burst_stays_quiet(self):
        rows = np.arange(1000, 1005)
        pages = burst_aware_pages(rows, np.zeros(5, int), 2, np.array([1e-3, 1e-3]),
                                  np.array([5.0, 5.0]), A=1e12)
        self.assertEqual(pages, [])


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.ref, cls.rng = fitted()

    def test_fit_learns_discrete_support(self):
        c = self.contract
        self.assertTrue(c.discrete[2])
        self.assertFalse(c.discrete[0])
        self.assertEqual(sorted(c.support[2].tolist()), [0.0, 1.0, 2.0, 3.0])

    def test_conformal_fit_sets_threshold_and_version(self):
        self.assertTrue(math.isfinite(self.contract.kappa))
        self.assertTrue(self.contract.version)

    def test_clean_stream_has_almost_no_flags(self):
        X = synthetic(10000, np.random.default_rng(11))
        flag = self.contract.check(X).flag.mean()
        self.assertLess(flag, 0.01)

    def test_magnitude_fault_is_flagged(self):
        X = synthetic(2000, np.random.default_rng(12))
        X[100, 0] = 1e9
        res = self.contract.check(X)
        self.assertTrue(res.flag[100])
        self.assertEqual(res.top[100, 0], 0)

    def test_unseen_category_is_a_hard_violation(self):
        X = synthetic(2000, np.random.default_rng(13))
        X[50, 2] = 9.0
        res = self.contract.check(X)
        self.assertTrue(res.flag[50])
        self.assertIn((50, 2), set(zip(res.hard_rows.tolist(), res.hard_cols.tolist())))

    def test_broken_relation_is_caught_with_plausible_values(self):
        X = synthetic(2000, np.random.default_rng(14))
        lo, hi = np.percentile(self.ref[:, 1], [20, 80])
        X[200, 0], X[200, 1] = 80.0, lo       # each value is common, the pair is not
        with_rel = self.contract.check(X)
        without = self.contract.check(X, use_relations=False)
        self.assertTrue(with_rel.flag[200])
        self.assertFalse(without.flag[200])

    def test_check_is_row_independent(self):
        X = synthetic(1500, np.random.default_rng(15))
        X[10, 1] = -500.0
        full = self.contract.check(X)
        alone = self.contract.check(X[10:11])
        self.assertEqual(bool(full.flag[10]), bool(alone.flag[0]))

    def test_chunking_does_not_change_results(self):
        X = synthetic(3000, np.random.default_rng(16))
        X[::97, 1] *= 1000.0
        a = self.contract.check(X, chunk=500)
        b = self.contract.check(X, chunk=20000)
        np.testing.assert_array_equal(a.flag, b.flag)
        np.testing.assert_array_equal(a.top, b.top)

    def test_missing_value_in_never_missing_feature_is_flagged(self):
        X = synthetic(1000, np.random.default_rng(17))
        X[7, 3] = np.nan
        self.assertTrue(self.contract.check(X).flag[7])

    def test_single_score_option_fits_and_scores(self):
        c, ref, _ = fitted(calibration="conformal1")
        X = synthetic(2000, np.random.default_rng(18))
        s_clean = c.score(X)
        X[5, 0] = 1e9
        self.assertTrue(math.isfinite(c.q))
        self.assertGreater(c.score(X)[5], s_clean[5])

    def test_without_relations_no_residual_threshold(self):
        rng = np.random.default_rng(0)
        ref = synthetic(25000, rng)
        c = FeatureContract(groups=[[0, 1]], alpha=1e-3, relations=False,
                            calibration="conformal").fit(ref[:20000], ref[20000:])
        self.assertEqual(c.t, math.inf)
        self.assertEqual(c.groups, [])


class RateCheckTests(unittest.TestCase):
    def test_missing_burst_raises_alarm(self):
        rng = np.random.default_rng(21)
        ref = synthetic(20000, rng)
        rc = RateCheck().fit(ref[:15000], ref[15000:])
        X = synthetic(2000, rng)
        w = rc.w
        X[w:2 * w, 1] = np.nan                  # a whole window of missing values
        kinds = {(j, k) for _, j, k in rc.alarms(X)}
        self.assertIn((1, "missing"), kinds)

    def test_stuck_value_raises_repeat_alarm(self):
        rng = np.random.default_rng(22)
        ref = synthetic(20000, rng)
        rc = RateCheck().fit(ref[:15000], ref[15000:])
        X = synthetic(2000, rng)
        X[:, 3] = 0.5
        kinds = {(j, k) for _, j, k in rc.alarms(X)}
        self.assertIn((3, "repeat"), kinds)

    def test_clean_stream_is_quiet(self):
        rng = np.random.default_rng(23)
        ref = synthetic(20000, rng)
        rc = RateCheck().fit(ref[:15000], ref[15000:])
        self.assertEqual(rc.alarms(synthetic(2000, rng)), [])

    def test_stream_shorter_than_a_window_has_no_alarms(self):
        rc = RateCheck()
        rc.thr = {"missing": np.full(4, 1.0), "repeat": np.full(4, 1.0)}
        self.assertEqual(rc.alarms(np.zeros((rc.w - 1, 4))), [])


class ConfigTests(unittest.TestCase):
    def test_grids_are_sorted(self):
        for g in (CONFIG["kappa_grid"], CONFIG["t_grid"]):
            self.assertEqual(list(g), sorted(g))

    def test_every_paper_fault_is_a_known_fault(self):
        self.assertTrue(set(CONFIG["paper_faults"]) <= set(CONFIG["faults"]))

    def test_every_listed_fault_has_an_operator(self):
        rng = np.random.default_rng(0)
        X = synthetic(500, rng)
        ref = reference_stats(X)
        for f in CONFIG["faults"]:
            inject(X, f, [1], 0.1, np.random.default_rng(1), ref)


if __name__ == "__main__":
    unittest.main()
