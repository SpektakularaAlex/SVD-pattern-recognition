# Handwritten Digit Classification with SVD Pattern Recognition
**Scientific Computing · Numerical Linear Algebra · Python**

This project implements a handwritten digit classifier using Singular Value Decomposition (SVD).  
The goal is to explore how numerical linear algebra can be applied to pattern recognition by constructing subspaces for each digit (0–9) and classifying test images based on projection residuals. All computations are performed using efficient NumPy matrix operations.

---

## Overview

Each handwritten digit (28×28 image) is reshaped into a 784-dimensional vector.  
Using a training set for each digit, the algorithm:

1. Builds a matrix whose columns are training images of a given digit  
2. Computes the SVD of this matrix  
3. Uses the first *k* singular vectors to form a digit “basis”  
4. Classifies test digits by projecting them onto each basis and measuring residuals  
5. Assigns the digit with the **smallest residual**

This approach is simple, interpretable, and surprisingly effective.

---

## Methods

### **1. Singular Value Decomposition (SVD)**  
For each digit, training images are stacked column-wise into a matrix `A` of size:


The classifier predicts the digit whose basis gives the smallest residual.

To handle all 40,000 test images efficiently, projection and residuals are computed using **vectorized matrix multiplications**, not loops.

---

## Results

### **Singular Images**  
The first few singular vectors show the most characteristic features of each digit.  
Digits like “3” or “8” show clear structure, with the first vector containing the dominant shape.

### **Singular Values**  
The singular values drop sharply for all digits, indicating that only a small number of basis vectors (`k = 5…15`) are needed for good approximations.

### **Classification Accuracy**  
Using 400 training images per digit and varying `k`:

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



