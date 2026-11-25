"""Malformed inject calls must fail loudly instead of corrupting silently."""
import unittest

import numpy as np

from fcontracts.inject import inject, reference_stats


class InjectInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.ref = rng.normal(size=(400, 3))
        cls.stats = reference_stats(cls.ref)
        cls.window = cls.ref[:50].copy()
        cls.rng = np.random.default_rng(0)

    def call(self, prevalence=0.2, features=(1,)):
        return inject(self.window, "magnitude", features, prevalence,
                      self.rng, self.stats)

    def test_invalid_prevalence_is_rejected_with_a_clear_error(self):
        for name, value in (("zero", 0.0), ("negative", -0.3),
                            ("above_one", 1.5), ("nan", float("nan")),
                            ("text", "high"), ("none", None)):
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "prevalence must be a real number in"):
                    self.call(prevalence=value)

    def test_invalid_features_are_rejected_with_a_clear_error(self):
        for name, value in (("empty", []), ("negative", [-1]),
                            ("too_large", [3]), ("float", [1.5]),
                            ("bool", [True]), ("text", ["a"])):
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "feature index|at least one column"):
                    self.call(features=value)

    def test_rejected_calls_leave_the_window_untouched(self):
        before = self.window.copy()
        for kwargs in ({"prevalence": 0.0}, {"prevalence": 2.0},
                       {"features": []}, {"features": [-1]}, {"features": [3]}):
            with self.assertRaises(ValueError):
                self.call(**kwargs)
        np.testing.assert_array_equal(self.window, before)

    def test_boundary_and_numpy_integer_calls_still_work(self):
        for name, kwargs in (("full", {"prevalence": 1.0}),
                             ("tiny", {"prevalence": 0.01}),
                             ("np_int_feature", {"features": [np.int64(2)]}),
                             ("np_float_prevalence", {"prevalence": np.float64(0.5)})):
            with self.subTest(case=name):
                X, changed = self.call(**kwargs)
                self.assertEqual(X.shape, self.window.shape)
                self.assertEqual(changed.shape, self.window.shape)
                self.assertTrue(changed.any())


if __name__ == "__main__":
    unittest.main()
