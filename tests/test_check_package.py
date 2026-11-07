"""Isolated file-presence tests. No datasets or third-party imports needed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import check_package


class PackageCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def touch(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def complete_inventory(self):
        for name in check_package.ENTRY_POINTS + check_package.FREEZE_MANIFESTS:
            self.touch(name)

    def test_empty_checkout_lists_each_required_file(self):
        report = check_package.check(self.root)
        self.assertFalse(report["complete"])
        self.assertEqual({x["path"] for x in report["missing"]},
                         set(check_package.ENTRY_POINTS + check_package.FREEZE_MANIFESTS))
        self.assertEqual(report["checked_summary_datasets"], [])

    def test_complete_inventory_is_not_a_scientific_validation(self):
        self.complete_inventory()
        report = check_package.check(self.root)
        self.assertTrue(report["complete"])
        self.assertIn("File presence only", report["scope"])
        self.assertEqual(report["checked_summary_datasets"], [])

    def test_run_csv_requires_all_three_summary_companions(self):
        self.complete_inventory()
        self.touch("results_v3/cc_runs.csv")
        self.touch("results_v3/cc_clean.json")
        report = check_package.check(self.root)
        self.assertEqual(report["checked_summary_datasets"], ["cc"])
        self.assertEqual({x["path"] for x in report["missing"]},
                         {"results_v3/cc_clean_flags.npz", "results_v3/cc_clean_pages.json"})
        self.touch("results_v3/cc_clean_flags.npz")
        self.touch("results_v3/cc_clean_pages.json")
        self.assertTrue(check_package.check(self.root)["complete"])

    def test_directory_is_not_a_required_file(self):
        self.complete_inventory()
        path = self.root / "sensitivity.py"
        path.unlink()
        path.mkdir()
        self.assertFalse(check_package.check(self.root)["complete"])

    def test_json_cli_exit_code_and_no_writes(self):
        before = sorted(self.root.rglob("*"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = check_package.main(["--repo", str(self.root), "--json"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.getvalue())["complete"])
        self.assertEqual(sorted(self.root.rglob("*")), before)

    def test_cli_success_and_missing_directory_error(self):
        self.complete_inventory()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_package.main(["--repo", str(self.root)]), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            check_package.main(["--repo", str(self.root / "not-here")])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
