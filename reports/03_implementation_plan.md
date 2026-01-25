# Implementation Plan (Built Repository)

## Project goals

This repository implements the methodology described in the manuscript *“Equivariant Transition Matrices for Explainable Deep Learning: A Lie Group Linearization Approach”* and provides reproducible pipelines for:

1. A synthetic numerical example (Section 3.4) including Algorithm 1 and Algorithm 2.
2. MNIST experiments (Section 3.5) including CNN training, autograd generator estimation for rotations, transition-matrix estimation (baseline and equivariant), and evaluation.

The implementation emphasizes:

- deterministic seeding and configuration capture
- explicit artifact storage (matrices, figures, logs)
- numerical stability checks and tests

## Folder structure

Top-level directories (required by the project contract):

- `inputs/` – JSON input matrices extracted from the manuscript
- `inputs/mnist/` – datasets (MNIST raw IDX files)
- `outputs/` – all generated artifacts
- `outputs/logs/` – logs for every run
- `reports/` – intermediate reports

Substructure:

- `inputs/synthetic/` contains Appendix 1.1 matrices in JSON
- `outputs/synthetic/{matrices,figures,runs}/` contain synthetic artifacts
- `outputs/mnist/{matrices,figures,models,runs}/` contain MNIST artifacts

## Reproducibility and configuration

- Central runner: `run_all.py`
- Config files:
  - `configs/synthetic.yaml`
  - `configs/mnist.yaml`
- Global seed setting in `src/etm/utils.py` (`set_global_seed`) sets Python, NumPy, and PyTorch seeds.
- Each run logs system information and writes a manifest JSON under `outputs/*/runs/`.

## Mathematical conventions implemented

The manuscript defines `T ∈ R^{l×k}` and the global mapping:

\[ B \approx A T^\top \]

Internally, we frequently use:

- `W := T^T ∈ R^{k×l}`
- so that `B ≈ A W`.

This convention avoids repeated transposes when applying the mapping to many samples.

## Algorithms

### Algorithm 2 (Synthetic)

Implemented in `src/synthetic/core.py::estimate_generator_via_bridge`:

1. MDS reduction to 2D
2. Linear regression decoder (2D → original dimension)
3. Small SO(2) rotation by ε
4. Decode to obtain `A_rot`
5. Solve `A J_A^T ≈ (A_rot − A)/ε` using a pseudoinverse

### Algorithm 1 (Synthetic)

Implemented in `src/synthetic/core.py::solve_equivariant_T`:

- Builds stacked system using Kronecker products
- Solves via SVD pseudoinverse with truncation threshold τ

### MNIST pipeline

Implemented in `src/mnist/pipeline.py`:

- **FM**: CNN with penultimate dimension k=490 (`src/mnist/model.py`)
- **MM**: flattened pixels (l=784)
- **Generator estimation**: uses autograd JVP through a differentiable rotation operator (`src/mnist/rotate.py`, `src/mnist/generators.py`)
- **Transition matrices**:
  - baseline `T_old`: SVD pseudoinverse on Gram matrix `(A^T A)^+ A^T B`
  - equivariant `T_new`: LSQR on an implicit stacked operator (SVD-grounded iterative method)
- **Evaluation**: SSIM, PSNR, symmetry error, and robustness curves (`src/mnist/eval.py`)

## Testing and validation

- Unit tests under `tests/` check parsing of repeating decimals and synthetic shape consistency.
- Synthetic pipeline saves all intermediate matrices required by Appendices 1.3–1.5.
- Numerical artifacts are saved in JSON for replay/replot.
