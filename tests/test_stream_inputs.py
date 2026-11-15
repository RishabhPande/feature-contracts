"""Invalid stream inputs must not produce an all-clear or an all-zero score."""
import unittest

import numpy as np

from fcontracts.contracts import FeatureContract


class StreamInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        ref = rng.normal(size=(6000, 3))
        cls.contract = FeatureContract([], 1e-3, calibration="conformal1").fit(
            ref[:5000], ref[5000:])
        cls.stream = ref[:100].copy()
        cls.stream[0, 0] = 1e12

    def methods(self):
        return (self.contract.check, self.contract.score)

    def test_negative_chunk_is_rejected_not_silently_skipped(self):
        for method in self.methods():
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    method(self.stream, chunk=-1)

    def test_zero_chunk_is_rejected(self):
        for method in self.methods():
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    method(self.stream, chunk=0)

    def test_noninteger_chunks_are_rejected(self):
        for chunk in (1.5, 2.0, "10", None, np.nan, np.inf):
            for method in self.methods():
                with self.subTest(chunk=chunk, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        method(self.stream, chunk=chunk)

    def test_boolean_chunks_are_rejected(self):
        for chunk in (True, False, np.bool_(True), np.bool_(False)):
            for method in self.methods():
                with self.subTest(chunk=chunk, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        method(self.stream, chunk=chunk)

    def test_wrong_feature_counts_are_rejected(self):
        for width in (0, 1, 2, 4):
            for method in self.methods():
                with self.subTest(width=width, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, f"3 features; got {width}"):
                        method(np.zeros((5, width)))

    def test_empty_stream_still_validates_feature_count(self):
        for method in self.methods():
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "3 features; got 1"):
                    method(np.empty((0, 1)))

    def test_wrong_dimensions_are_rejected(self):
        for shape in ((), (3,), (2, 3, 1)):
            for method in self.methods():
                with self.subTest(shape=shape, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "two-dimensional NumPy array"):
                        method(np.zeros(shape))

    def test_nonarrays_are_rejected(self):
        for value in (None, [[1, 2, 3]], "not an array"):
            for method in self.methods():
                with self.subTest(value=value, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "two-dimensional NumPy array"):
                        method(value)

    def test_empty_valid_stream_has_correct_result_shapes(self):
        X = np.empty((0, 3))
        r = self.contract.check(X)
        self.assertEqual(r.n, 0)
        self.assertEqual(r.flag.shape, (0,))
        self.assertEqual(r.top.shape, (0, 3))
        for name in ("viol_rows", "viol_cols", "hard_rows", "hard_cols"):
            self.assertEqual(getattr(r, name).shape, (0,))
        self.assertEqual(self.contract.score(X).shape, (0,))

    def test_valid_chunks_preserve_flags_evidence_and_scores(self):
        ref = self.contract.check(self.stream)
        scores = self.contract.score(self.stream)
        self.assertTrue(ref.flag[0])
        for chunk in (1, 13, 1000, np.int64(17), np.uint32(23)):
            with self.subTest(chunk=chunk):
                r = self.contract.check(self.stream, chunk=chunk)
                for name in ("flag", "top", "viol_rows", "viol_cols", "hard_rows", "hard_cols"):
                    np.testing.assert_array_equal(getattr(r, name), getattr(ref, name))
                np.testing.assert_array_equal(self.contract.score(self.stream, chunk=chunk), scores)

    def test_sliced_noncontiguous_arrays_are_supported(self):
        X = self.stream[::2]
        self.assertFalse(X.flags.c_contiguous)
        np.testing.assert_array_equal(self.contract.check(X).flag,
                                      self.contract.check(X.copy()).flag)
        np.testing.assert_array_equal(self.contract.score(X), self.contract.score(X.copy()))

    def test_invalid_calls_do_not_mutate_the_fitted_contract(self):
        before = self.contract.check(self.stream).flag.copy()
        for method in self.methods():
            with self.assertRaises(ValueError):
                method(self.stream, chunk=-1)
        np.testing.assert_array_equal(before, self.contract.check(self.stream).flag)


if __name__ == "__main__":
    unittest.main()
