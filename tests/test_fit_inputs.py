"""Malformed fit slices must not damage an already usable contract."""
import pickle
import unittest

import numpy as np

from fcontracts.contracts import FeatureContract


class FitInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(98)
        cls.ref = rng.normal(size=(800, 3))
        cls.fit_slice = cls.ref[:600]
        cls.cal_slice = cls.ref[600:]

    def invalid_pairs(self):
        for name, value in (("none", None), ("list", [[1, 2, 3]]),
                            ("scalar", np.array(1)), ("vector", np.ones(3)),
                            ("cube", np.ones((2, 3, 1)))):
            for side in ("X_fit", "X_cal"):
                args = [self.fit_slice, self.cal_slice]
                args[side == "X_cal"] = value
                yield name + "_" + side, args, side + " must be a two-dimensional NumPy array"
        for shape in ((0, 3), (4, 0), (0, 0)):
            for side in ("X_fit", "X_cal"):
                args = [self.fit_slice, self.cal_slice]
                args[side == "X_cal"] = np.empty(shape)
                yield str(shape) + side, args, side + " must have at least one row and one feature"
        for width in (1, 2, 4):
            yield "width_" + str(width), [self.fit_slice, np.ones((4, width))], "X_fit and X_cal must have the same feature count"

    def test_invalid_slices_are_rejected_with_clear_errors(self):
        for calibration in ("grid", "conformal", "conformal1"):
            for name, args, message in self.invalid_pairs():
                with self.subTest(calibration=calibration, case=name):
                    c = FeatureContract([], 0.05, calibration=calibration)
                    with self.assertRaisesRegex(ValueError, message):
                        c.fit(*args)

    def test_invalid_refits_preserve_all_fitted_state_and_predictions(self):
        for calibration in ("grid", "conformal", "conformal1"):
            c = FeatureContract([[0, 1]], 0.05, calibration=calibration).fit(
                self.fit_slice, self.cal_slice)
            state = pickle.dumps(c.__dict__)
            before = c.check(self.ref)
            for name, args, message in self.invalid_pairs():
                with self.subTest(calibration=calibration, case=name):
                    with self.assertRaisesRegex(ValueError, message):
                        c.fit(*args)
                    self.assertEqual(pickle.dumps(c.__dict__), state)
                    after = c.check(self.ref)
                    for field in ("flag", "top", "viol_rows", "viol_cols", "hard_rows", "hard_cols"):
                        np.testing.assert_array_equal(getattr(after, field), getattr(before, field))

    def test_one_row_slices_are_supported(self):
        for calibration in ("grid", "conformal", "conformal1"):
            with self.subTest(calibration=calibration):
                c = FeatureContract([], 0.05, calibration=calibration).fit(
                    self.fit_slice[:1], self.cal_slice[:1])
                self.assertEqual(c.check(self.cal_slice[:1]).flag.shape, (1,))

    def test_noncontiguous_and_missing_value_slices_remain_supported(self):
        for calibration in ("grid", "conformal", "conformal1"):
            with self.subTest(calibration=calibration):
                X = self.ref.copy()
                X[::19, 0] = np.nan
                a, b = X[:600:2], X[600::2]
                self.assertFalse(a.flags.c_contiguous)
                c = FeatureContract([], 0.05, calibration=calibration).fit(a, b)
                d = FeatureContract([], 0.05, calibration=calibration).fit(a.copy(), b.copy())
                np.testing.assert_array_equal(c.check(X).flag, d.check(X).flag)


if __name__ == "__main__":
    unittest.main()
