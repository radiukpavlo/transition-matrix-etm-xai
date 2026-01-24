# Data Processing Pipeline and Methodology

This document provides a comprehensive step-by-step explanation of the current data processing pipeline, detailing how matrices are processed, the mathematical formulations involved, and the specific algorithms used for both Synthetic and MNIST experiments.

## 1. Synthetic Data Pipeline

The synthetic pipeline works with low-dimensional data to validate Algorithm 1 (Equivariant Solution) and Algorithm 2 (Generator Estimation) as described in the manuscript.

### Step 1: Data Loading & Preparation

**Input Matrices:**

* $A \in \mathbb{R}^{m \times k}$: Source "latent" states (rows are samples).
* $B \in \mathbb{R}^{m \times l}$: Target "observable" states.
* $T_{old}^{provided} \in \mathbb{R}^{k \times l}$: A provided baseline transition matrix (provided as transpose $W$ in the codebase).

**Processing:**
A naive least-squares baseline $T_{old, ls}$ is computed immediately to serve as a comparison point. This minimizes the fidelity term $\|B - AW\|_F^2$ without equivariance constraints:
$$ W_{old} = A^\dagger B $$
where $A^\dagger$ is the SVD-based pseudoinverse.

### Step 2: Generator Estimation (Algorithm 2)

This step estimates the infinitesimal generators $J^A$ and $J^B$ which represent the underlying symmetry (approximated as rotation) for the matrices $A$ and $B$, respectively.

**Process for a matrix $X$ (either $A$ or $B$):**

1. **Dimensionality Reduction**: Map the high-dimensional matrix $X$ to a 2D manifold using Multidimensional Scaling (MDS):
    $$ Z = \text{MDS}(X) \in \mathbb{R}^{m \times 2} $$

2. **Decoder Fitting**: Train a linear regression decoder $D: \mathbb{R}^2 \to \mathbb{R}^{\text{dim}(X)}$ such that $D(Z) \approx X$.

3. **Perturbation**: Rotate the 2D embeddings $Z$ by a small angle $\epsilon$. The rotation matrix $R_{\epsilon}$ is defined as:
    $$ R_{\epsilon} = \begin{bmatrix} \cos \epsilon & -\sin \epsilon \\ \sin \epsilon & \cos \epsilon \end{bmatrix} $$
    The rotated embeddings are:
    $$ Z_{\epsilon} = Z R_{\epsilon}^T $$

4. **Decoding**: Map the rotated embeddings back to the original high-dimensional space using the fitted decoder:
    $$ X_{\epsilon} = D(Z_{\epsilon}) $$

5. **Finite Difference**: Compute the approximate tangent vectors (velocities):
    $$ \Delta = \frac{X_{\epsilon} - X}{\epsilon} $$

6. **Generator Solution**: Solve for the generator $J$ satisfying the Lie algebra relation $\dot{X} = X J^T$:
    $$ J^T = X^\dagger \Delta \implies J = (X^\dagger \Delta)^T $$

### Step 3: Equivariant Transition Matrix (Algorithm 1)

The pipeline solves for the transition matrix $T \in \mathbb{R}^{l \times k}$ (where $B \approx A T^T$) that respects the estimated symmetries.

**Objective Function:**
$$ \min_T \underbrace{\| B - A T^T \|_F^2}_{\text{Fidelity}} + \lambda \underbrace{\| T J^A - J^B T \|_F^2}_{\text{Symmetry Error}} $$

**Vectorization & Solution:**
The optimization problem is rewritten as a linear system $M \text{vec}(T) = y$ using Kronecker products to solve it explicitly:

1. **Fidelity Term**: Maps to $(A \otimes I_l) \text{vec}(T) \approx \text{vec}(B^T)$.
2. **Symmetry Term**: Maps to $((J^A)^T \otimes I_l - I_k \otimes J^B) \text{vec}(T) \approx 0$.
3. **Stacked System**:
    $$ M = \begin{bmatrix} A \otimes I_l \\ \lambda ((J^A)^T \otimes I_l - I_k \otimes J^B) \end{bmatrix} $$
    $$ y = \begin{bmatrix} \text{vec}(B^T) \\ 0 \end{bmatrix} $$

The system is solved using the SVD-based pseudoinverse: $\text{vec}(T) = M^\dagger y$.

**Lambda Sweep**:
The solution is computed for multiple values of $\lambda$ to analyze the trade-off between the Mean Squared Error (Fidelity) and the Symmetry Error.

### Step 4: Robustness Analysis (Scenario 3)

To test robustness, the latent representations are rotated by various angles $\alpha$. Predictions are made using both the baseline $T_{old}$ and the equivariant $T_{new}$:
$$ B^*_{old} = A(\alpha) T_{old}^T $$
$$ B^*_{new} = A(\alpha) T_{new}^T $$
These predictions are compared against the "true" rotated target $B(\alpha)$ (obtained via the decoder) to quantify how well the transition matrix preserves the manifold structure under transformation.

---

## 2. MNIST Data Pipeline

The MNIST pipeline adapts the methodology for high-dimensional image data ($l = 784$) and neural network features ($k = \text{feature dim}$), where explicit Kronecker product solutions are computationally infeasible.

### Step 1: Extraction & Generator Estimation

Instead of MDS, the pipeline uses a trained Convolutional Neural Network (CNN) and pixel-level rotations.

**Generators via Finite Differences:**
The pipeline estimates generators $J^A$ (feature space) and $J^B$ (pixel space) using central finite differences on the rotation group $SO(2)$.

1. **Perturbation**: For a batch of images $X$, create $X_{+\epsilon}$ (rotated by $+\epsilon$) and $X_{-\epsilon}$ (rotated by $-\epsilon$).
2. **Feature Extraction**: Pass the original and rotated images through the CNN to get features $A, A_{+\epsilon}, A_{-\epsilon}$.
3. **Derivative Approximation**:
    $$ dA \approx \frac{A_{+\epsilon} - A_{-\epsilon}}{2\epsilon}, \quad dB \approx \frac{X_{+\epsilon} - X_{-\epsilon}}{2\epsilon} $$
4. **Least Squares Estimation**:
    Solve for $J^A$ and $J^B$ over the collected samples:
    $$ J^A = ((A^T A)^\dagger A^T dA)^T $$
    $$ J^B = ((B^T B)^\dagger B^T dB)^T $$

### Step 2: Transition Matrix Optimization (LSQR)

Due to the size of $B$ ($m \times 784$) and the Kronecker product dimension ($kl \times kl$), the explicit matrix solution from Algorithm 1 is infeasible.

**Implicit Solver:**
The pipeline uses `scipy.sparse.linalg.lsqr` with a `LinearOperator`. It iteratively minimizes the same objective as Synthetic Step 3 without forming the matrix $M$.

**Operator Definition:**
For a candidate vector $w = \text{vec}(W)$ (where $W=T^T$):

* **Forward Map (MatVec)**: Computes residuals for Fidelity and Symmetry.
    $$ r_1 = \text{vec}(AW - B) $$
    $$ r_2 = \sqrt{\lambda} \cdot \text{vec}((J^A)^T W - W (J^B)^T) $$
    *Note: The implementation uses the identity $(J^A)^T W - W (J^B)^T \approx 0$, which is the transpose of the Synthetic relation $T J^A - J^B T = 0$.*

* **Adjoint Map (RMatVec)**: Computes the gradient update direction from the residuals.
    $$ g = A^T r_1 + \sqrt{\lambda} (J^A r_2 - r_2 J^B) $$

### Step 3: Experimentation

* **Lambda Sweep**: The LSQR solver is run for multiple $\lambda$ values.
* **Evaluation**: The learned $T_{new}$ relates feature space dynamics to pixel space dynamics. The quality is measured by applying feature rotations and checking if the predicted image rotates accordingly (evaluating MSE on rotated test sets).

---

## 3. Summary of Matrices

| Matrix | Dimensions | Source | Description |
| :--- | :--- | :--- | :--- |
| **$A$** | $m \times k$ | `inputs/` or CNN Features | Source state (features/latent). |
| **$B$** | $m \times l$ | `inputs/` or Images | Target state (observations/pixels). |
| **$J^A$** | $k \times k$ | Algorithm 2 (MDS+LinReg) / Finite Diff | Infinitesimal generator on $A$. |
| **$J^B$** | $l \times l$ | Algorithm 2 (MDS+LinReg) / Finite Diff | Infinitesimal generator on $B$. |
| **$T$** | $l \times k$ | Algorithm 1 / LSQR | The transition operator ($B \approx A T^T$). |
| **$W$** | $k \times l$ | $T^T$ | The transpose of $T$, often stored as `T_kxl.json`. |
