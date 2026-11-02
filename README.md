# Linear Algebra Studio

A desktop application for exploring core linear algebra operations through an interactive matrix editor and step-by-step row-reduction output.

Built with **Python**, **PyQt6**, and **NumPy**, the project brings common matrix calculations into one focused workspace.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt&logoColor=white)
![Numerical library](https://img.shields.io/badge/Numerical%20computing-NumPy-013243?logo=numpy&logoColor=white)

---

## Overview

Linear Algebra Studio provides a graphical interface for entering matrices, running common operations, and reviewing the results. The underlying algorithms use elementary row operations, while their intermediate steps are printed to the terminal for learning and inspection.

The project was originally developed as a Linear Algebra course project and is maintained as a standalone educational application.

## Features

### System of linear equations
- Enter the number of equations and variables.
- Build and edit an augmented matrix directly in the application.
- Solve supported systems using Gauss–Jordan elimination.
- Identify inconsistent systems and systems with infinitely many solutions.

### Square matrix operations
- Calculate the inverse of a square matrix using row operations.
- Calculate a determinant while accounting for row swaps.
- Start with an identity matrix as a convenient input template.

### Desktop experience
- Modern dark and light appearances.
- Dedicated workflows for systems of equations and square matrices.
- Adjustable matrix dimensions, with support for dimensions from 1 to 20.
- Recent-matrix history with load, delete, and clear actions.
- Inline results with a copy-to-clipboard action.
- Scrollable matrix editing area for larger inputs.
- Input validation and clear error messages.

### Educational output
The numerical routines print their intermediate steps to the terminal, including row swaps, pivot normalization, and elimination operations. This makes it possible to follow how the input matrix is transformed.

## Getting started

### Requirements

- Python 3.10 or newer
- NumPy
- PyQt6

### Installation

Clone the repository:

```bash
git clone https://github.com/ArvinNotDev/Linear_Algebra.git
cd Linear_Algebra
```

Create and activate a virtual environment (recommended):

**Windows — PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python main.py
```

## Using the application

1. Choose **System of equations** or **Square matrix** from the workflow selector.
2. Set the required dimensions and select **Apply size**.
3. Enter your coefficients or matrix values.
4. Run the relevant operation:
   - **Solve system** for a supported system of equations.
   - **Inverse** to calculate a matrix inverse.
   - **Determinant** to calculate the determinant.
5. Review the result in the application. Use the history panel to reload a previously processed matrix.

The history is kept in memory for the current application session and holds up to 12 entries.

## Project structure

```text
Linear_Algebra/
├── main.py          # PyQt6 desktop application and matrix editor
├── functions.py     # Gauss–Jordan, inverse, and determinant routines
├── requirements.txt # Python dependencies
└── README.md
```

## Implementation notes

- The equation solver currently expects a square coefficient system: the number of equations must match the number of variables.
- Matrix dimensions are limited to 1–20 in the graphical interface.
- The numerical routines use floating-point arithmetic, so results may contain small rounding differences.
- Intermediate calculation steps are written to the terminal where the application was launched.
- Matrix history is session-only; it is not written to disk.

## License

No license has been specified for this repository. Please contact the repository owner before redistributing or reusing the project beyond applicable default copyright permissions.
