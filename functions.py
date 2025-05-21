import numpy as np

def print_matrix(M, step_description=""):
    print(f"\n{step_description}")
    for row in M:
        print("  ".join(f"{val:8.3f}" for val in row))
    print()

def swap_rows(M, i):
    for k in range(i + 1, len(M)):
        if M[k, i] != 0:
            M[[i, k]] = M[[k, i]]
            print(f"\nSwapped row {i} with row {k}")
            print_matrix(M, f"After swapping row {i} with row {k}")
            return True
    return False
def gauss_jordan(M):
    M = M.astype(float)
    rows, cols = M.shape
    n = rows
    print_matrix(M, "Initial Augmented Matrix:")

    for i in range(n):
        if M[i, i] == 0:
            swapped = False
            for k in range(i + 1, n):
                if M[k, i] != 0:
                    M[[i, k]] = M[[k, i]]
                    print(f"\nSwapped row {i} with row {k}")
                    print_matrix(M, f"After swapping row {i} with row {k}")
                    swapped = True
                    break

            if not swapped:
                if np.all(M[i, :-1] == 0) and M[i, -1] == 0:
                    print(f"\nRow {i} is all zeros ⇒ part of infinite solutions")
                    continue
                elif np.all(M[i, :-1] == 0) and M[i, -1] != 0:
                    print(f"\nInconsistent row at {i} ⇒ no solution")
                    return "no solution"

        pivot = M[i, i]
        if pivot == 0:
            continue

        M[i] = M[i] / pivot
        print(f"\nNormalized row {i} by pivot {pivot}")
        print_matrix(M, f"After making pivot in row {i} = 1")

        for j in range(n):
            if j != i:
                factor = M[j, i]
                M[j] = M[j] - factor * M[i]
                print(f"Eliminated row {j} using row {i} (factor: {factor})")
                print_matrix(M, f"After eliminating row {j}")

    rank = np.linalg.matrix_rank(M[:, :-1])
    augmented_rank = np.linalg.matrix_rank(M)

    if rank < n:
        print("\nSystem has infinite solutions")
        return "infinity"
    elif rank < augmented_rank:
        print("\nSystem is inconsistent")
        return "no solution"

    solution = M[:, -1]
    print_matrix(M, "Final RREF Matrix:")
    return solution


def invert_matrix(A):
    A = A.astype(float)
    n = A.shape[0]
    I = np.identity(n, dtype=float)

    print_matrix(A, "Original Matrix A:")
    print_matrix(I, "Identity Matrix I:")

    for i in range(n):
        if A[i, i] == 0:
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    I[[i, k]] = I[[k, i]]
                    print(f"\nSwapped row {i} with row {k}")
                    print_matrix(A, f"A after swapping row {i} and {k}")
                    print_matrix(I, f"I after swapping row {i} and {k}")
                    break
            else:
                raise ValueError("Matrix is not invertible.")

        pivot = A[i, i]
        if pivot == 0:
            raise ValueError("Matrix is not invertible.")

        A[i] = A[i] / pivot
        I[i] = I[i] / pivot

        print(f"\nNormalized row {i} by pivot {pivot}")
        print_matrix(A, "A after normalization")
        print_matrix(I, "I after normalization")

        for j in range(n):
            if j != i:
                factor = A[j, i]
                if factor != 0:
                    A[j] = A[j] - factor * A[i]
                    I[j] = I[j] - factor * I[i]
                    print(f"\nEliminated row {j} using row {i} (factor {factor})")
                    print_matrix(A, "A after elimination")
                    print_matrix(I, "I after elimination")

    print_matrix(A, "Final A (should be identity):")
    print_matrix(I, "Final I (should be inverse):")

    if not np.allclose(A, np.identity(n)):
        raise ValueError("Matrix is not invertible.")

    return I

def det_by_row_ops(A):
    A = A.astype(float)
    n = A.shape[0]
    det = 1
    swap_count = 0

    print_matrix(A, "Original Matrix for Determinant:")

    if np.all(A == 0):
        print("\nMatrix is entirely zero ⇒ determinant is 0")
        return 0

    for i in range(n):
        if A[i, i] == 0:
            swapped = False
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    swap_count += 1
                    swapped = True
                    print(f"\nSwapped row {i} with row {k}")
                    print_matrix(A, f"After swapping row {i} with row {k}")
                    break
            if not swapped:
                print(f"\nColumn {i} has no pivot ⇒ determinant is 0")
                return 0

        pivot = A[i, i]
        det *= pivot
        A[i] = A[i] / pivot

        for j in range(i + 1, n):
            factor = A[j, i]
            A[j] = A[j] - factor * A[i]
            print(f"\nEliminated row {j} using row {i} (factor: {factor})")
            print_matrix(A, f"After eliminating row {j}")

    det *= (-1) ** swap_count
    print(f"\nDeterminant: {det:.3f}")
    return det
