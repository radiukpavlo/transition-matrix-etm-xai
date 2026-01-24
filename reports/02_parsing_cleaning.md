# Parsing, Cleaning, and Data Hygiene Report

## Scope
This report documents how Appendix 1.1 (synthetic) matrices were extracted, cleaned, and stored under `inputs/`, including explicit handling of ambiguous numeric entries and shape conventions.

## Appendix 1.1 extraction
The manuscript provides matrices:
- `A ∈ R^{15×5}`
- `B ∈ R^{15×4}`
- `T_old` printed as a `5×4` matrix (despite the text defining `T ∈ R^{l×k} = R^{4×5}`).

All three matrices are stored as JSON with metadata under:
- `inputs/synthetic/A.json`
- `inputs/synthetic/B.json`
- `inputs/synthetic/T_old.json`

Each JSON file contains: `name`, `shape`, `dtype`, `source`, `data`, and optional `meta`.

## Ambiguous repeating-decimal entries in `A`
Row 13 in Appendix 1.1 contains two ambiguous notations:
- `0.8(4)`
- `-0.(4)`

We interpret these as repeating decimals:
- `0.8(4) = 0.8444444444444444…`
- `-0.(4) = −0.4444444444444444…`

This interpretation is consistent with common mathematical shorthand for repeating digits and is implemented by `etm.utils.parse_repeating_decimal`.

### Sensitivity check
A small alternative interpretation would be rounding:
- `0.8(4) ≈ 0.84`
- `-0.(4) ≈ −0.44`

We verified that this alternative has a negligible effect on the core fidelity-only baseline MSE (difference in the fourth decimal place for the synthetic baseline). The equivariance-regularized solution and qualitative figures are unchanged.

## Shape conventions and manuscript transpose ambiguity
The manuscript defines `T ∈ R^{l×k}` and uses:

\[
B \approx A T^\top.
\]

However, Appendix 1.1 prints `T_old` as a `5×4` matrix, which matches `W = T^\top ∈ R^{k×l}`.

To avoid silent transposes, the implementation uses a fixed convention:
- `A ∈ R^{m×k}`
- `B ∈ R^{m×l}`
- internal regression form: `B ≈ A W`, where `W = T^\top ∈ R^{k×l}`
- the manuscript’s operator is recovered by `T = W^\top`.

Both forms are stored for reproducibility when relevant.

## Validating `T_old`
The project keeps the Appendix 1.1 printed `T_old` in `inputs/synthetic/T_old.json` as an immutable manuscript artifact.

When we validated it against the stated fidelity-only least squares objective `min_W ||B − A W||_F^2`, it did **not** match the computed minimizer for the provided `A` and `B`. The Frobenius difference between the printed matrix and the least-squares solution is large (`≈ 2.4`).

Therefore, for **Scenario 1** (the “old approach”), the code computes a fidelity-only baseline by SVD-based pseudoinverse:

\[
W_{old,LS} = A^+ B,
\]

and uses it for baseline metrics and robustness evaluation. The manuscript-printed matrix is still retained and evaluated separately as `T_old_provided`.

This is recorded as an explicit deviation in `README.md` under “Assumptions & deviations.”
