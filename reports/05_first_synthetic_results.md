# Synthetic Results (Section 3.4)

## 1. Experimental setup
- Samples: `m = 15` (3 classes, 5 samples each).
- FM features: `A ∈ R^{15×5}`.
- MM features: `B ∈ R^{15×4}`.
- Symmetry group: `SO(2)` (one generator, `r = 1`).

Algorithm 2 is used to estimate generators `J^A` and `J^B` by:
1. `MDS(A) → A_2D`, `MDS(B) → B_2D` (2D embeddings).
2. Training linear decoders `A_2D→A` and `B_2D→B`.
3. Rotating the 2D embeddings by `ε = 0.01` radians and decoding back.
4. Solving `A J_A^T ≈ (A_rot − A)/ε` and `B J_B^T ≈ (B_rot − B)/ε`.

Algorithm 1 then computes the transition matrix `T_new` by minimizing:

\[
\mathcal{L}(T)=\|B^T - T A^T\|_F^2 + \lambda \|T J^A - J^B T\|_F^2.
\]

## 2. Key artifacts
All matrices are stored in JSON format under:
- Inputs: `inputs/synthetic/`
- Outputs: `outputs/synthetic/matrices/`

Figures are stored under:
- `outputs/synthetic/figures/`

## 3. Metrics (Scenario 1 vs Scenario 2)
We report the manuscript-defined metrics:
- `MSE_fid = ||B - B*||_F^2 /(m·l)`
- `Sym_err = ||T J^A - J^B T||_F^2`

**Baseline (old approach, fidelity-only):** we use the least-squares solution `W_old = argmin_W ||B - A W||_F^2` computed via an SVD-based pseudoinverse.

**Equivariant (new approach):** Algorithm 1 with `λ = 0.5`.

| Metric | Old Approach (Fidelity-only) | New Approach (Equivariant, λ=0.5) |
|---|---:|---:|
| MSE on training data | 0.003670 | 0.005467 |
| Symmetry defect (Sym_err) | 13166.993 | 0.045063 |

The equivariant solution sacrifices a small amount of fidelity while reducing symmetry defect by ~5 orders of magnitude.

## 4. λ sweep (fidelity–equivariance trade-off)
The sweep `λ ∈ {0, 0.1, 0.25, 0.5, 1, 2}` yields:

| λ | MSE_fid | Sym_err |
|---:|---:|---:|
| 0.00 | 0.003670 | 13166.993 |
| 0.10 | 0.005432 | 0.146623 |
| 0.25 | 0.005461 | 0.048664 |
| 0.50 | 0.005467 | 0.045063 |
| 1.00 | 0.005474 | 0.044319 |
| 2.00 | 0.005550 | 0.042476 |

See figures:
- `08_tradeoff_mse_vs_lambda.png`
- `09_tradeoff_sym_vs_lambda.png`

## 5. Robustness test (Scenario 3)
We generate rotated test matrices `A_rot(α)` for angles `α ∈ [−30°, +30°]` with 5° step using the MDS-bridge rotation.
We form `B_target(α)` by applying the analogous rotation-and-decoding procedure in the MM space.

We evaluate rotated-data error `MSE_rot(α) = ||B_target(α) - B*(α)||_F^2 /(m·l)` for both methods.

Summary:
- Old method average rotated MSE: ≈ 0.00373
- New method average rotated MSE: ≈ 0.00369

See:
- `10_robustness_scatter_old_vs_new.png` (author-requested qualitative comparison)

## 6. Notes on manuscript-provided `T_old`
Appendix 1.1 prints a matrix labeled `T_old` (shape 5×4). When applied to the Appendix 1.1 matrices `A` and `B`, it does **not** solve the stated fidelity-only least squares problem (its training MSE is 0.03998 vs 0.00367 for the SVD-based solution).

To remain faithful to the manuscript equations (Scenario 1 defines a fidelity-only optimum), we therefore:
- **store** the printed matrix as an input artifact (`inputs/synthetic/T_old.json`), and
- **use** the computed least-squares baseline for the Scenario 1 comparison.

A dedicated discrepancy discussion is included in `reports/parsing_cleaning.md`.
