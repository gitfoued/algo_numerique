import numpy as np
import time
import matplotlib.pyplot as plt
import ctypes

# Chargement de la bibliothèque partagée Fortran
fortran_lib = fortran_lib = ctypes.CDLL('C:\\Users\\alama\\OneDrive\\Documents\\projet algo-numerique\\Fortran+python\\calculation.so')


# Définition des types pour les arguments et les valeurs de retour des sous-programmes Fortran
fortran_lib.generate_positive_definite_matrix.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
fortran_lib.solve_with_gauss.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
fortran_lib.solve_with_FLU.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
fortran_lib.solve_with_cholesky.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]

# Définition du type de retour des sous-programmes Fortran
fortran_lib.generate_positive_definite_matrix.restype = None
fortran_lib.solve_with_gauss.restype = ctypes.c_double
fortran_lib.solve_with_FLU.restype = ctypes.c_double
fortran_lib.solve_with_cholesky.restype = ctypes.c_double

def generate_positive_definite_matrix(size):
    A = np.zeros((size, size), dtype=np.float64)
    fortran_lib.generate_positive_definite_matrix(ctypes.c_int(size), A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return A

def solve_with_gauss(matrix, trials=5):
    n = matrix.shape[0]
    total_time = 0.0
    for _ in range(trials):
        start_time = time.time()
        fortran_lib.solve_with_gauss(matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(n), None)
        total_time += time.time() - start_time
    return total_time / trials

def solve_with_FLU(matrix, trials=5):
    n = matrix.shape[0]
    total_time = 0.0
    for _ in range(trials):
        start_time = time.time()
        fortran_lib.solve_with_FLU(matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(n), None)
        total_time += time.time() - start_time
    return total_time / trials

def solve_with_cholesky(matrix, trials=5):
    n = matrix.shape[0]
    total_time = 0.0
    for _ in range(trials):
        start_time = time.time()
        fortran_lib.solve_with_cholesky(matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(n), None)
        total_time += time.time() - start_time
    return total_time / trials

def measure_execution_time(algorithm, matrix, trials=5):
    total_time = 0
    for _ in range(trials):
        total_time += algorithm(matrix)
    return total_time / trials

def main():
    matrix_sizes = [100]
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
