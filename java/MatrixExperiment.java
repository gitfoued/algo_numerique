import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.WindowConstants;

public class MatrixExperiment {
    public static void main(String[] args) {
        // Définir les tailles de matrices à tester
        int[] matrixSizes = {100, 400, 500, 700, 1000, 1500, 2000};
        int trials = 5; // Nombre d'essais par test

        // Initialiser les tableaux pour stocker les temps de chaque algorithme
        long[][] timesGauss = new long[matrixSizes.length][trials];
        long[][] timesFLU = new long[matrixSizes.length][trials];
        long[][] timesCholesky = new long[matrixSizes.length][trials];

        // Pour chaque taille de matrice
        for (int i = 0; i < matrixSizes.length; i++) {
            int size = matrixSizes[i];
            final int indexI = i;
            // Générer et résoudre la matrice avec chaque algorithme
            double[][] matrix = generatePositiveDefiniteMatrix(size);
            
            for (int j = 0; j < trials; j++) {
                double[][] matrixCopy = copyMatrix(matrix);

                timesGauss[i][j] = measureExecutionTime(() -> solveWithGauss(matrixCopy), 1);
                matrixCopy = copyMatrix(matrix);
                timesFLU[i][j] = measureExecutionTime(() -> solveWithFLU(matrixCopy), 1);
                matrixCopy = copyMatrix(matrix);
                timesCholesky[i][j] = measureExecutionTime(() -> solveWithCholesky(matrixCopy), 1);
            }
        }

        // Afficher les résultats sous forme de tableau dans une fenêtre Swing
        displayResultsTable(matrixSizes, timesGauss, timesFLU, timesCholesky);
    }

    // Méthode pour afficher les résultats sous forme de tableau dans une fenêtre Swing
    public static void displayResultsTable(int[] matrixSizes, long[][] timesGauss, long[][] timesFLU, long[][] timesCholesky) {
        // Créer un tableau de données pour les résultats
        Object[][] data = new Object[matrixSizes.length][4];
        for (int i = 0; i < matrixSizes.length; i++) {
            int size = matrixSizes[i];
            long gaussAvg = average(timesGauss[i]);
            long fluAvg = average(timesFLU[i]);
            long choleskyAvg = average(timesCholesky[i]);
            data[i] = new Object[]{size, gaussAvg, fluAvg, choleskyAvg};
        }

        // Créer un tableau pour afficher les données
        String[] columnNames = {"Matrix Size", "Gauss Time (ms)", "FLU Time (ms)", "Cholesky Time (ms)"};
        JTable table = new JTable(data, columnNames);

        // Créer un panneau pour contenir le tableau
        JPanel panel = new JPanel();
        panel.add(new JScrollPane(table));

        // Créer une fenêtre pour afficher le panneau
        JFrame frame = new JFrame("Matrix Experiment Results");
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    // Méthode pour calculer la moyenne d'un tableau de long
    public static long average(long[] arr) {
        long sum = 0;
        for (long num : arr) {
            sum += num;
        }
        return sum / arr.length;
    }

    // Méthode pour générer une matrice symétrique définie positive
    public static double[][] generatePositiveDefiniteMatrix(int size) {
        Random random = new Random();
        double[][] matrix = new double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j <= i; j++) {
                double value = random.nextDouble();
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
        return matrix;
    }

    // Méthode pour copier une matrice
    public static double[][] copyMatrix(double[][] original) {
        int n = original.length;
        double[][] copy = new double[n][n];

        for (int i = 0; i < n; i++) {
            System.arraycopy(original[i], 0, copy[i], 0, n);
        }
        return copy;
    }

    // Méthode pour mesurer le temps d'exécution d'une tâche
    public static long measureExecutionTime(Runnable task, int trials) {
        long totalTime = 0;

        for (int i = 0; i < trials; i++) {
            long startTime = System.nanoTime();
            task.run();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime) / 1_000_000; // Convertir des nanosecondes en millisecondes
        }
        return totalTime / trials;
    }

   // Implémentation de l'algorithme de résolution Gaussienne
   public static void solveWithGauss(double[][] matrix) {
    int n = matrix.length;
    double[] solution = new double[n];

    // Étape de l'élimination gaussienne
    for (int i = 0; i < n; i++) {
        for (int k = i + 1; k < n; k++) {
            double factor = matrix[k][i] / matrix[i][i];
            for (int j = i; j < n + 1; j++) {
                matrix[k][j] -= factor * matrix[i][j];
            }
        }
    }

    // Résolution du système triangulaire supérieur
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += matrix[i][j] * solution[j];
        }
        solution[i] = (matrix[i][n] - sum) / matrix[i][i];
    }

    // Affichage de la solution
    System.out.println("Solution using Gauss:");
    for (int i = 0; i < n; i++) {
        System.out.println("x[" + i + "] = " + solution[i]);
    }
}

// Implémentation de l'algorithme de résolution FLU (Factorisation LU)
public static void solveWithFLU(double[][] matrix) {
    int n = matrix.length;
    double[][] L = new double[n][n];
    double[][] U = new double[n][n];
    double[] y = new double[n];
    double[] solution = new double[n];

    // Factorisation LU
    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;

        for (int k = i; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * U[j][k];
            }
            U[i][k] = matrix[i][k] - sum;
        }

        for (int k = i + 1; k < n; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L[k][j] * U[j][i];
            }
            L[k][i] = (matrix[k][i] - sum) / U[i][i];
        }
    }

    // Résolution du système triangulaire Ly = b
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (matrix[i][n] - sum) / L[i][i];
    }

    // Résolution du système triangulaire Ux = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += U[i][j] * solution[j];
        }
        solution[i] = (y[i] - sum) / U[i][i];
    }

    // Affichage de la solution
    System.out.println("Solution using FLU:");
    for (int i = 0; i < n; i++) {
        System.out.println("x[" + i + "] = " + solution[i]);
    }
}

// Implémentation de l'algorithme de résolution Cholesky
public static void solveWithCholesky(double[][] matrix) {
    int n = matrix.length;
    double[][] L = new double[n][n];
    double[] y = new double[n];
    double[] x = new double[n];

    // Décomposition de Cholesky
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += Math.pow(L[j][k], 2);
                }
                L[j][j] = Math.sqrt(matrix[j][j] - sum);
            } else {
                for (int k = 0; k < j; k++) {
                    sum += (L[i][k] * L[j][k]);
                }
                L[i][j] = (matrix[i][j] - sum) / L[j][j];
            }
        }
    }

    // Résolution du système triangulaire Ly = b
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (matrix[i][n] - sum) / L[i][i];
    }

    // Résolution du système triangulaire L^T x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    // Affichage de la solution
    System.out.println("Solution using Cholesky:");
    for (int i = 0; i < n; i++) {
        System.out.println("x[" + i + "] = " + x[i]);
    }
}
}
