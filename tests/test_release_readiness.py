from __future__ import annotations

import unittest

from omega_self_check import run_self_check


class ReleaseReadinessTests(unittest.TestCase):
    def test_quick_release_self_check(self) -> None:
        result = run_self_check(full_tests=False, demo_years=5)
        self.assertEqual(result["software_status"], "READY")
        self.assertEqual(result["checks_failed"], 0)
        self.assertGreaterEqual(result["checks_passed"], 7)


if __name__ == "__main__":
    unittest.main()
