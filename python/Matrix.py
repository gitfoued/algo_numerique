import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import lu, cholesky

def generate_positive_definite_matrix(size):
    np.random.seed(0)
    A = np.random.rand(size, size)
    A = np.dot(A, A.T)  # Assurer que la matrice est symétrique définie positive
    return A

def solve_with_gauss(matrix, trials=5):  # Ajout de trials comme paramètre pour le nb de test
    total_time = 0
    for _ in range(trials):
        start_time = time.time()
        n = len(matrix)
        for i in range(n):
            max_val = np.abs(matrix[i, i])
            max_row = i
            for k in range(i + 1, n):
                if np.abs(matrix[k, i]) > max_val:
                    max_val = np.abs(matrix[k, i])
                    max_row = k
            if max_row != i:
                matrix[[i, max_row]] = matrix[[max_row, i]]
            for k in range(i + 1, n):
                factor = matrix[k, i] / matrix[i, i]
                matrix[k, i:] -= factor * matrix[i, i:]
        solution = np.linalg.solve(matrix[:, :-1], matrix[:, -1])
        total_time += time.time() - start_time
    return total_time / trials

def solve_with_FLU(matrix, trials=5):  # Ajout de trials comme paramètre
    total_time = 0
    for _ in range(trials):
        start_time = time.time()
        A = matrix[:, :-1]
        b = matrix[:, -1]
        P, L, U = lu(A)
        y = np.linalg.solve(L, P @ b)
        solution = np.linalg.solve(U, y)
        total_time += time.time() - start_time
    return total_time / trials

def solve_with_cholesky(matrix, trials=5):  # Ajout de trials comme paramètre
    total_time = 0
    for _ in range(trials):
        start_time = time.time()
        try:
            L = cholesky(matrix[:, :-1])
            b = matrix[:, -1]
            y = np.linalg.solve(L, b)
            solution = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            continue  # Passer à la prochaine itération si la matrice n'est pas positive définie
        total_time += time.time() - start_time
    return total_time / trials

def measure_execution_time(algorithm, matrix, trials=5):
    total_time = 0
    for _ in range(trials):
        total_time += algorithm(matrix)
    return total_time / trials

def main():
    matrix_sizes = [100, 400, 500, 700, 1000, 1500, 2000]
    gauss_times = []
    flu_times = []
    cholesky_times = []
    trials = 5  # Nombre d'essais par test

    for size in matrix_sizes:
        matrix = generate_positive_definite_matrix(size)
        matrix = np.column_stack((matrix, np.random.rand(size)))  # Ajouter la colonne des valeurs b

        gauss_time = measure_execution_time(solve_with_gauss, matrix, trials)
        flu_time = measure_execution_time(solve_with_FLU, matrix, trials)
        cholesky_time = measure_execution_time(solve_with_cholesky, matrix, trials)

        gauss_times.append(gauss_time)
        flu_times.append(flu_time)
        cholesky_times.append(cholesky_time)

    # Plotting
    plt.plot(matrix_sizes, gauss_times, label='Gauss')
    plt.plot(matrix_sizes, flu_times, label='FLU')
    plt.plot(matrix_sizes, cholesky_times, label='Cholesky')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Direct Methods for Solving Positive Definite Symmetric Matrices')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
