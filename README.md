
# 📘 Linear Algebra Toolkit (Python)

This repository is a temporary submission for showcasing a **Linear Algebra project** to **Dr. Hosseinzadeh**. It provides implementations of key matrix operations using Python and NumPy, designed for clarity, traceability, and educational use.

---

## 🔧 Features

- ✅ **Gauss-Jordan Elimination**  
  Solves systems of linear equations and detects inconsistent or infinite solution cases.

- ✅ **Matrix Inversion**  
  Computes the inverse of a square matrix using row operations.

- ✅ **Determinant Calculation**  
  Calculates the determinant via row-reduction, accounting for row swaps.

- ✅ **Verbose Output**  
  Every step (row swaps, normalizations, eliminations) is printed in detail to help users understand the transformation process.

- ✅ **Graphical Interface with PyQt**  
  Includes a simple GUI built with PyQt for interactive input and visualization.

---

## 🧪 Example Output

You will see console outputs like:

```
Initial Augmented Matrix:
   2.000    1.000    5.000
   4.000    4.000   12.000

Normalized row 0 by pivot 2.0
After making pivot in row 0 = 1:
   1.000    0.500    2.500
   4.000    4.000   12.000

Eliminated row 1 using row 0 (factor: 4.0)
After eliminating row 1:
   1.000    0.500    2.500
   0.000    2.000    2.000
```

---

## 📂 Files

- `main.py` — Contains:
  - `gauss_jordan(M)`
  - `invert_matrix(A)`
  - `det_by_row_ops(A)`
  - `print_matrix()` utility

---

## 🧠 Educational Goals

This toolkit is created for **demonstration and understanding**, not for high-performance computation. Each algorithm is written with clarity in mind, mimicking manual row operations taught in linear algebra course in **AUSMT**.

---

## ▶️ Getting Started

### Requirements

- Python 3.x
- NumPy

### Run Example

```bash
pip install numpy
python main.py
```

You can modify matrices inside `main.py` to test different operations.

---

## 🧑‍🏫 Instructor

Project submitted to: **Dr. Hosseinzadeh**

---

## 📝 License

This project is for academic and demonstrative purposes only.
