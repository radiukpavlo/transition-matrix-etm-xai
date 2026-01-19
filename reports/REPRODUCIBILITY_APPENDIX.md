# Reproducibility Appendix

This appendix provides a complete, runnable protocol to reproduce the results of
the manuscript's **equivariant transition matrix** method and its **baseline
transition matrix** comparator, following the experimental structure in the
baseline PDF.

Throughout, we use the manuscript's convention:
- Deep/latent representation: $a(x)\in\mathbb{R}^k$ (for MNIST: $k=490$)
- Interpretable/mental representation: $b(x)\in\mathbb{R}^{\ell}$ (for MNIST: $\ell=784$)
- Transition matrix: $T\in\mathbb{R}^{\ell\times k}$ mapping $a\mapsto b$ so that
  $b(x)\approx T a(x)$.

For a dataset of $m$ paired samples, define matrices
$A\in\mathbb{R}^{m\times k}$ and $B\in\mathbb{R}^{m\times \ell}$ by stacking rows
$A_{j,:}=a(x_j)^\top$ and $B_{j,:}=b(x_j)^\top$.

---

## 1. Reproduction overview and scope

This appendix covers two experiments, exactly matching the manuscript:

1) **Synthetic experiment (Section 3.4):**
   - Uses the explicit $A\in\mathbb{R}^{15\times 5}$, $B\in\mathbb{R}^{15\times 4}$,
     and baseline $T_{\text{old}}$ values given in the manuscript.
   - Compares baseline vs. equivariant $T_{\text{new}}$ using:
     - Fidelity mean-squared error $\mathrm{MSE}_{\mathrm{fid}}$
     - Symmetry defect $\mathrm{Sym}_{\mathrm{err}}$
   - Performs a rotation-robustness stress test by randomly rotating each sample
     and evaluating error against an “ideal rotated” target.

2) **MNIST experiment (Section 3.5):**
   - Reproduces the baseline PDF structure:
     - Train a small CNN on full MNIST.
     - Extract penultimate-layer features ($k=490$).
     - Define mental-model features as pixels ($\ell=784$).
     - Fit the baseline transition matrix on a random subset of $m=10{,}000$
       training images.
     - Evaluate reconstructions on $1{,}000$ test images using SSIM and PSNR.
   - Applies the manuscript’s equivariant objective with SO(2) rotation
     generators estimated by finite differences.

This appendix also includes:
- A solver-equivalence proof (vectorized SVD formulation vs. implicit CG solver).
- Memory and runtime estimates justifying why Kronecker matrices are not formed
  for MNIST.
- Expected output artifacts and verification checks.

---

## 2. Data and experimental setup

### 2.1 Synthetic experiment inputs

Use the exact matrices provided in the manuscript:
- $A\in\mathbb{R}^{15\times 5}$ (deep features)
- $B\in\mathbb{R}^{15\times 4}$ (mental features)
- $T_{\text{old}}\in\mathbb{R}^{5\times 4}$ (baseline transition matrix, in the
  baseline-PDF convention $AT\approx B$)

There are $3$ classes with $5$ points each:
- class 0: rows 1–5
- class 1: rows 6–10
- class 2: rows 11–15

### 2.2 MNIST data

- Dataset: MNIST handwritten digits.
- Training set: 60,000 images; test set: 10,000 images.
- Images: 28×28 grayscale.

Preprocessing (used everywhere in the scripts):
- Convert to float and normalize to $[0,1]$ by dividing raw uint8 values by 255.
- Flatten pixels into $b(x)\in\mathbb{R}^{784}$ when constructing $B$.

Subsampling (baseline PDF structure, reused for both baseline and equivariant methods):
- Choose a random subset of **$m=10{,}000$ training images** to build $A$ and $B$.
- Choose a random subset of **$1{,}000$ test images** for evaluation.

Randomness control:
- Fix an explicit global seed for:
  - NumPy RNG (subsampling)
  - Torch RNG (CNN initialization and training)

### 2.3 CNN architecture for MNIST (deep feature map $a$)

Use the CNN architecture described in the baseline PDF (penultimate layer size $k=490$):

- **conv_block_1**
  - Conv2D: in=1, out=10, kernel=3, stride=1, padding=1
  - ReLU
  - Conv2D: in=10, out=10, kernel=3, stride=1, padding=1
  - ReLU
  - MaxPool2D: kernel=2, stride=2
- **conv_block_2**
  - Conv2D: in=10, out=10, kernel=3, stride=1, padding=1
  - ReLU
  - Conv2D: in=10, out=10, kernel=3, stride=1, padding=1
  - ReLU
  - MaxPool2D: kernel=2, stride=2
- Flatten ($10\times 7\times 7 = 490$)
- Linear: 490 → 10

Define $a(x)$ as the **flattened 490‑dim vector** immediately before the final linear classifier.

---

## 3. Baseline method (as implemented in the PDF)

### 3.1 Baseline objective and closed-form solution

The baseline PDF estimates a transition matrix by solving a linear least-squares
problem. In the baseline-PDF convention:

\[
\min_{T_{\text{pdf}}\in\mathbb{R}^{k\times \ell}} \|A T_{\text{pdf}} - B\|_F^2.
\]

The manuscript uses $T\in\mathbb{R}^{\ell\times k}$ with
$B^\top\approx T A^\top$, which is equivalent under

\[
T = T_{\text{pdf}}^\top.
\]

Given $A\in\mathbb{R}^{m\times k}$ and $B\in\mathbb{R}^{m\times \ell}$, the
baseline solution is

\[
T_{\text{pdf}} = A^+ B,
\]

where $A^+$ is the Moore–Penrose pseudoinverse.

The baseline PDF specifies two computation routes:

1) If $A$ has full column rank ($\det(A^\top A)\neq 0$):
\[
A^+ = (A^\top A)^{-1} A^\top.
\]

2) Otherwise, compute $A^+$ by SVD with truncation:
If $A = U\Sigma V^\top$, then
\[
A^+ = V \Sigma^+ U^\top,
\]
where $\Sigma^+$ inverts only singular values above a numerical threshold.

Our reference implementation follows this logic and returns the manuscript-shaped
matrix $T\in\mathbb{R}^{\ell\times k}$.

### 3.2 Baseline reconstruction

Given a test deep feature matrix $A_{\mathrm{test}}\in\mathbb{R}^{m_{\mathrm{test}}\times k}$,
reconstruct mental features (pixels) as
\[
\widehat{B}_{\mathrm{test}} = A_{\mathrm{test}} T^\top \in\mathbb{R}^{m_{\mathrm{test}}\times \ell}.
\]

Reshape each row of $\widehat{B}_{\mathrm{test}}$ to 28×28 to form a reconstructed image.

### 3.3 Baseline evaluation on MNIST (SSIM and PSNR)

For each test image $I$ (ground truth) and $\widehat{I}$ (reconstruction), the
baseline PDF uses:

- **SSIM** (Structural Similarity Index Measure)
- **PSNR** (Peak Signal-to-Noise Ratio)

The scripts compute both per-image and then aggregate mean/median and
distribution plots.

---

## 4. Equivariant method (as implemented in the manuscript)

### 4.1 Objective

The manuscript introduces an equivariance regularizer for a Lie group action.
For $r$ generator directions, estimate $T\in\mathbb{R}^{\ell\times k}$ by

\[
\min_T\; \underbrace{\|B^\top - T A^\top\|_F^2}_{\text{fidelity}} +
\lambda \sum_{i=1}^r \underbrace{\|T J_i^A - J_i^B T\|_F^2}_{\text{symmetry defect}}.
\]

In the experiments provided, $r=1$ and the Lie group is SO(2) rotation.

### 4.2 Vectorized least-squares form

Using the identity $\mathrm{vec}(X Y Z) = (Z^\top\otimes X)\,\mathrm{vec}(Y)$, we can write:

- Fidelity term:
  \[
  \mathrm{vec}(T A^\top) = (A\otimes I_{\ell})\,\mathrm{vec}(T).
  \]

- Symmetry term for generator pair $(J^A, J^B)$:
  \[
  \mathrm{vec}(T J^A - J^B T) = \big((J^A)^\top \otimes I_{\ell} - I_k\otimes J^B\big)\,\mathrm{vec}(T).
  \]

Define
\[
K = (J^A)^\top \otimes I_{\ell} - I_k\otimes J^B.
\]

Then the objective is equivalent to the stacked least-squares problem

\[
\min_u \left\|\begin{bmatrix}A\otimes I_{\ell}\\ \sqrt{\lambda}\,K\end{bmatrix} u -
\begin{bmatrix}\mathrm{vec}(B^\top)\\ 0\end{bmatrix}\right\|_2^2,
\quad u = \mathrm{vec}(T).
\]

### 4.3 Implementation note on the $\lambda$ weight

The manuscript text defines the objective with a factor $\lambda$ multiplying the
symmetry term. In a stacked least-squares implementation, this corresponds to a
block weight of $\sqrt{\lambda}$.

In code, we therefore implement the objective exactly as written by weighting the
symmetry block with $\sqrt{\lambda}$.

---

## 5. Generator estimation procedure

This section fully specifies how $J^A$ and $J^B$ are computed in both experiments.

### 5.1 General finite-difference definition

Fix a small scalar $\varepsilon>0$ and a Lie algebra direction $\xi$.
For each sample $x_j$ define a perturbed sample

\[
\widetilde{x}_j = \exp(\varepsilon\xi)\cdot x_j.
\]

Compute finite differences in both representations:
\[
\Delta a_j = \frac{a(\widetilde{x}_j) - a(x_j)}{\varepsilon},\qquad
\Delta b_j = \frac{b(\widetilde{x}_j) - b(x_j)}{\varepsilon}.
\]

Stack these as matrices $\Delta A\in\mathbb{R}^{m\times k}$ and
$\Delta B\in\mathbb{R}^{m\times \ell}$.

Estimate generators by least squares:
\[
\Delta A \approx A (J^A)^\top, \qquad \Delta B \approx B (J^B)^\top.
\]

In practice we use a ridge-stabilized solve (Tikhonov regularization):
\[
(J^A)^\top = (A^\top A + \mu I)^{-1} A^\top \Delta A,
\qquad
(J^B)^\top = (B^\top B + \mu I)^{-1} B^\top \Delta B.
\]

The scripts report the conditioning of $A^\top A$ and $B^\top B$ and the chosen $\mu$.

### 5.2 Synthetic generator estimation (Algorithm 2 bridge)

The synthetic experiment uses the manuscript’s 2D “bridge” construction:

1. Apply MDS (multidimensional scaling) to map high-dimensional points to 2D:
   - $A\mapsto A_{2d}\in\mathbb{R}^{m\times 2}$
   - $B\mapsto B_{2d}\in\mathbb{R}^{m\times 2}$

2. Learn a linear decoder from 2D back to the original space:
   - $\widehat{A} = A_{2d} W_A + c_A$
   - $\widehat{B} = B_{2d} W_B + c_B$

3. Rotate the 2D points by $\varepsilon$ using the standard 2D rotation matrix
   $R(\varepsilon)$.

4. Decode rotated points back into the original spaces to obtain
   $A^{\varepsilon}$ and $B^{\varepsilon}$.

5. Compute finite differences $\Delta A$ and $\Delta B$ and solve for $J^A, J^B$ as above.

### 5.3 MNIST generator estimation (SO(2) rotations of images)

For MNIST, the group action is an **in-plane rotation of the image** about its
center.

Procedure:

1. Choose a small angle $\varepsilon$ (in radians). In code we implement rotation
   by converting to degrees and applying a standard 2D image rotation with
   bilinear interpolation.

2. For each selected generator-estimation sample $x_j$:
   - Create $\widetilde{x}_j = \mathrm{Rotate}(x_j, +\varepsilon)$
   - Compute deep features $a(x_j)$ and $a(\widetilde{x}_j)$
   - Use $b(x_j)$ as the flattened pixel vector and $b(\widetilde{x}_j)$ as the
     flattened rotated pixel vector

3. Compute $\Delta A$ and $\Delta B$ and solve the ridge least squares problems
   for $J^A$ and $J^B$.

Interpolation and padding:
- Use constant padding (background value 0) to avoid wraparound.
- Use bilinear interpolation for the default setting.

---

## 6. Solver details and equivalence proof

This section proves that the large-scale CG solver (which never forms Kronecker
products) solves the same optimization problem as the explicit vectorized SVD
solver.

### 6.1 Vectorized normal equations

Consider the single-generator objective
\[
\min_T\; \|B^\top - T A^\top\|_F^2 + \lambda\|T J^A - J^B T\|_F^2.
\]

Define $u=\mathrm{vec}(T)$, $y=\mathrm{vec}(B^\top)$, and
\[
M = \begin{bmatrix}A\otimes I_{\ell}\\ \sqrt{\lambda}\,K\end{bmatrix},
\quad K = (J^A)^\top\otimes I_{\ell} - I_k\otimes J^B.
\]

Then the objective equals $\|Mu - [y;0]\|_2^2$. Any minimizer satisfies the
(normal) equations
\[
(M^\top M)u = M^\top\begin{bmatrix}y\\ 0\end{bmatrix}.
\]

Using Kronecker identities,
\[
(A\otimes I_{\ell})^\top (A\otimes I_{\ell}) = (A^\top A)\otimes I_{\ell},
\]
so
\[
M^\top M = (A^\top A)\otimes I_{\ell} + \lambda K^\top K.
\]
Also
\[
(A\otimes I_{\ell})^\top y = \mathrm{vec}(B^\top A).
\]

Therefore the solution vector satisfies
\[
\Big((A^\top A)\otimes I_{\ell} + \lambda K^\top K\Big)\,\mathrm{vec}(T) = \mathrm{vec}(B^\top A).
\]

### 6.2 Matrix-form linear operator (no Kronecker matrices)

Let $G=A^\top A\in\mathbb{R}^{k\times k}$ and $C=B^\top A\in\mathbb{R}^{\ell\times k}$.

Define the symmetry residual $S(T) = T J^A - J^B T\in\mathbb{R}^{\ell\times k}$.
Then one can show that
\[
K^\top K\,\mathrm{vec}(T) = \mathrm{vec}\big( S(T) (J^A)^\top - (J^B)^\top S(T) \big).
\]

Hence the normal equations are equivalent to the matrix equation
\[
T G + \lambda\big( S(T) (J^A)^\top - (J^B)^\top S(T) \big) = C.
\]

This expression involves only ordinary matrix products of sizes
$\ell\times k$, $k\times k$, and $\ell\times \ell$.

### 6.3 Equivalence to the implemented CG solver

The CG solver in this project defines a linear operator $\mathcal{L}$ acting on
$T$ by
\[
\mathcal{L}(T) = T G + \lambda\big( S(T) (J^A)^\top - (J^B)^\top S(T) \big) + \gamma T,
\]
where $\gamma\ge 0$ is an optional tiny ridge added only to ensure strict
positive definiteness.

It then solves
\[
\mathcal{L}(T) = C
\]
by running conjugate gradient on $\mathrm{vec}(T)$.

Because $\mathcal{L}$ is exactly the matrix-form expansion of
$(A^\top A)\otimes I_{\ell} + \lambda K^\top K$ (plus optional $\gamma I$), the
CG solution is algebraically identical to the minimizer of the original
least-squares objective.

---

## 7. Hyperparameters and sensitivity analysis

### 7.1 Synthetic experiment settings

- Rotation direction: SO(2)
- Generator estimation angle: $\varepsilon = 0.01$ radians
- Equivariance weight: $\lambda = 0.5$
- Ridge for generator estimation: $\mu = 10^{-6}$ (small, only for numerical stability)
- SVD truncation threshold for the stacked system: $\tau=10^{-12}$ (relative to $\sigma_{\max}$)

### 7.2 MNIST experiment settings

CNN training:
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 128
- Epochs: configurable (default 3 in the scripts)

Transition matrix estimation:
- Training subset size: $m=10{,}000$
- Test subset size: $1{,}000$
- Baseline pseudoinverse: normal-equation solve when $A^\top A$ is well-conditioned,
  otherwise SVD pseudoinverse.

Equivariant method:
- Generator estimation subset size: configurable (default 2,000 samples)
- Generator estimation angle: sweep $\varepsilon\in\{0.005,0.01,0.02\}$ radians
- Equivariance weight sweep: $\lambda\in\{0, 10^{-3}, 10^{-2}, 5\times 10^{-2}, 10^{-1}, 5\times 10^{-1}, 1\}$
- CG stopping: relative tolerance 1e-5 (configurable)
- Optional ridge $\gamma$: 0 by default; set to 1e-8–1e-6 if CG encounters numerical issues.

### 7.3 Reported sensitivity outputs

The scripts generate:
- A $\lambda$ trade-off curve (fidelity MSE vs. symmetry defect)
- A table logging generator conditioning as a function of $\varepsilon$

---

## 8. Runtime and memory estimates

This section explains why the MNIST solver must avoid explicit Kronecker matrices.

### 8.1 Dimensions (MNIST)

- $m = 10{,}000$ (training subset)
- $k = 490$ (CNN penultimate)
- $\ell = 784$ (pixels)
- Unknowns: $\ell k = 784\times 490 = 384{,}160$

### 8.2 Naïve Kronecker storage is infeasible

The fidelity block in the vectorized system is
$M_{\mathrm{fid}} = A\otimes I_{\ell}$ with shape
$(m\ell)\times (k\ell)$.

- Rows: $m\ell = 10{,}000\times 784 = 7{,}840{,}000$
- Cols: $k\ell = 490\times 784 = 384{,}160$
- Entries: $\approx 3.01\times 10^{12}$

Even at float32 (4 bytes), storing $M_{\mathrm{fid}}$ would require
$\approx 12$ TB of RAM, which is infeasible.

### 8.3 Feasible implicit storage

The implicit (matrix-form) solver stores only:
- $A$: $10{,}000\times 490$ float32 ≈ 19.6 MB
- $B$: $10{,}000\times 784$ float32 ≈ 31.4 MB
- $T$: $784\times 490$ float32 ≈ 1.5 MB
- $J^A$: $490\times 490$ float32 ≈ 0.96 MB
- $J^B$: $784\times 784$ float32 ≈ 2.35 MB
- Gram matrix $G=A^\top A$: $490\times 490$ float64/float32 negligible

Total working set is comfortably below a few hundred MB even with temporary
buffers.

### 8.4 Time complexity (dominant operations)

Per CG iteration, the dominant costs are dense matrix multiplications involving
$T$ with $G$, $J^A$, and $J^B$.

The scripts therefore:
- Use a limited iteration budget
- Use the baseline solution as warm-start
- Provide a $\lambda$ sweep on a reduced subset (optional)

---

## 9. Expected output artifacts and verification checks

All outputs are written under `outputs/`.

### 9.1 Synthetic expected artifacts

- `outputs/synthetic/synthetic_metrics.json`
  - Contains:
    - `mse_fid_old`, `mse_fid_new`
    - `sym_err_old`, `sym_err_new`
    - `mse_rot_old`, `mse_rot_new`

- `outputs/synthetic/synthetic_scatter_rotated_old_vs_new.png`
  - Side-by-side 2D scatter of rotated predictions under $T_{\text{old}}$ vs $T_{\text{new}}$.

Verification check (qualitative):
- $T_{\text{new}}$ should preserve clearer class structure under random rotations
  than $T_{\text{old}}$.

### 9.2 MNIST expected artifacts

After MNIST data are available and `scripts/run_mnist.py` is executed:

- `outputs/mnist/metrics_baseline_unrotated.csv`
- `outputs/mnist/metrics_equivariant_unrotated.csv`
- `outputs/mnist/metrics_baseline_rotated.csv`
- `outputs/mnist/metrics_equivariant_rotated.csv`

- `outputs/mnist/mnist_baseline_recon_examples.png`
- `outputs/mnist/mnist_equivariant_recon_examples.png`

- `outputs/mnist/mnist_tradeoff_lambda.png`

Verification checks:
- Baseline reconstructions should yield nontrivial SSIM/PSNR values.
- The equivariant method should reduce the symmetry defect
  $\|T J^A - J^B T\|_F$ and improve robustness on rotated test images for suitable $\lambda$.

---

## 10. Code organization and execution instructions

Project layout:

- `src/data/mnist_idx.py` — MNIST IDX download + parsing (no torchvision)
- `src/models/cnn_mnist.py` — CNN architecture with 490‑dim penultimate features
- `src/methods/transition_baseline.py` — baseline pseudoinverse transition matrix
- `src/methods/generator_estimation.py` — generator estimation (synthetic + MNIST)
- `src/methods/transition_equivariant.py` — equivariant solvers (small SVD + large CG)
- `src/utils/metrics.py` — fidelity + symmetry defect + SSIM/PSNR
- `src/utils/plotting.py` — figure creation

Entry points:

- Synthetic (offline):
  ```bash
  python scripts/run_synthetic.py --outdir outputs/synthetic
  ```

- MNIST (requires data):
  ```bash
  python scripts/run_mnist.py --download --outdir outputs/mnist
  ```

- Run everything:
  ```bash
  python scripts/run_all.py
  ```

The scripts write a `run_metadata.json` containing seeds, platform info, and
package versions to support full provenance tracking.
