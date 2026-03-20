"""Tests for OOD evaluation signature and availability."""

from __future__ import annotations

import unittest
from evaluation.ood_eval import evaluate_ood

class TestOODEval(unittest.TestCase):
    def test_evaluate_ood_callable(self) -> None:
        self.assertTrue(callable(evaluate_ood))

if __name__ == "__main__":
    unittest.main()
