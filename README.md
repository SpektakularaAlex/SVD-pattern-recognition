
# Handwritten Digit Classification with SVD Pattern Recognition
**Scientific Computing · Numerical Linear Algebra · Python**

📄 **Full report:** [BerVet_miniproject_2.pdf](report/BerVet_miniproject_2.pdf)

This project classifies handwritten digits (0–9) by building a low-dimensional subspace for each digit using **SVD**, then predicting labels via **projection residuals**. 
The goal is to explore how numerical linear algebra can be applied to pattern recognition by constructing subspaces for each digit and classifying test images based on projection residuals. All computations are performed using efficient NumPy matrix operations.

---

## Overview

Each 28×28 image is reshaped into a 784-dimensional vector. For each digit \(d\), we stack \(n\) training vectors into a matrix
\[
A_d \in \mathbb{R}^{784 \times n}.
\]
We compute
\[
A_d = U_d \Sigma_d V_d^\top,
\]
and use the first \(k\) left singular vectors \(U_{d,k}\) as a basis for digit \(d\).

To classify a test image \(\delta\), we project onto each digit basis and compute the residual
\[
r_d(\delta) = \lVert (I - U_{d,k}U_{d,k}^\top)\delta \rVert_2.
\]
The predicted digit is the one with **smallest residual**.

The implementation is vectorized to handle all test images efficiently.

---

## Results

### Singular images (examples)
<p align="center">
  <img src="figures/singular_images_digit3.png" width="700">
</p>

<p align="center">
  <img src="figures/singular_images_digit8.png" width="700">
</p>

### Singular values (examples)
<p align="center">
  <img src="figures/singular_values_digit3.png" width="700">
</p>

<p align="center">
  <img src="figures/singular_values_digit8.png" width="700">
</p>

### Accuracy vs number of basis vectors \(k\)
<p align="center">
  <img src="figures/accuracy_per_digit_vs_k.png" width="700">
</p>

Using **400 training images per digit** and **40,000 test images**, the total accuracy increases with \(k\) and saturates around \(k \approx 10\).

| k | Accuracy (%) |
|---|--------------|
| 5 | 91.81 |
| 6 | 92.33 |
| 7 | 93.02 |
| 8 | 93.41 |
| 9 | 93.88 |
| 10 | 94.00 |
| 11 | 94.14 |
| 12 | 94.36 |
| 13 | 94.53 |
| 14 | 94.65 |
| 15 | 94.70 |

Increasing `k` improves accuracy but with diminishing returns beyond ~10 basis vectors.

Digits with consistent shapes (e.g., “1”) achieve very high accuracy.  
Digits with high variation (e.g., “8”) are harder to classify.


---

## Reproducibility

### Requirements
- Python 3.x
- NumPy, Matplotlib

### Run
1. Place the provided `.npy` datasets in the project directory (or update paths).
2. Run the main script / notebook to reproduce figures and accuracy table.

