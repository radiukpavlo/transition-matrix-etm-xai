import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from etm.utils import load_json_matrix, parse_repeating_decimal
from etm.synthetic import solve_equivariant_T, mse_fid, sym_err


class TestSynthetic(unittest.TestCase):
    def test_repeating_decimal_parser(self):
        self.assertAlmostEqual(parse_repeating_decimal("0.8(4)"), 0.8444444444444444)
        self.assertAlmostEqual(parse_repeating_decimal("-0.(4)"), -0.4444444444444444)

    def test_lam0_matches_fidelity_least_squares(self):
        repo = Path(__file__).resolve().parents[1]
        A = load_json_matrix(repo / "inputs" / "synthetic" / "A.json")
        B = load_json_matrix(repo / "inputs" / "synthetic" / "B.json")

        # Least squares baseline for B ≈ A W
        W_ls, *_ = np.linalg.lstsq(A, B, rcond=None)
        mse_ls = mse_fid(B, A @ W_ls)

        # Algorithm 1 with λ=0 should reduce to the same fidelity-only LS problem.
        JA = np.eye(A.shape[1])
        JB = np.eye(B.shape[1])
        T_lxk, _M, _meta = solve_equivariant_T(A, B, [JA], [JB], lam=0.0, tau=1e-12)
        W = T_lxk.T
        mse_alg = mse_fid(B, A @ W)

        self.assertAlmostEqual(mse_alg, mse_ls, places=10)
        self.assertLess(sym_err(T_lxk, JA, JB), 1e-10)


if __name__ == "__main__":
    unittest.main()
