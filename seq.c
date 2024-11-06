#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_POINTS 10000000   // Ajuste para um milhão de pontos
#define NUM_DIMENSIONS 10    // Ajuste de 10 dimensões
#define K 5
#define MAX_ITERATIONS 100

// Função para calcular a distância euclidiana entre dois pontos
double euclidean_distance(double *a, double *b, int dimensions) {
    double distance = 0.0;
    for (int i = 0; i < dimensions; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

// Função para o algoritmo k-means na versão paralela otimizada
void kmeans_parallel(double points[NUM_POINTS][NUM_DIMENSIONS], int labels[NUM_POINTS], double centroids[K][NUM_DIMENSIONS]) {
    int iterations = 0;
    while (iterations < MAX_ITERATIONS) {
        int changes = 0;

        // Atribui cada ponto ao centróide mais próximo
        for (int i = 0; i < NUM_POINTS; i++) {
            int nearest_centroid = 0;
            double min_distance = euclidean_distance(points[i], centroids[0], NUM_DIMENSIONS);

            for (int j = 1; j < K; j++) {
                double distance = euclidean_distance(points[i], centroids[j], NUM_DIMENSIONS);
                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_centroid = j;
                }
            }

            if (labels[i] != nearest_centroid) {
                labels[i] = nearest_centroid;
                changes++;
            }
        }

        // Acumula os novos centróides em variáveis privadas
        double new_centroids[K][NUM_DIMENSIONS] = {0};
        int counts[K] = {0};

        {
            double local_centroids[K][NUM_DIMENSIONS] = {0};
            int local_counts[K] = {0};

            for (int i = 0; i < NUM_POINTS; i++) {
                int cluster = labels[i];
                local_counts[cluster]++;
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    local_centroids[cluster][d] += points[i][d];
                }
            }

            {
                for (int j = 0; j < K; j++) {
                    counts[j] += local_counts[j];
                    for (int d = 0; d < NUM_DIMENSIONS; d++) {
                        new_centroids[j][d] += local_centroids[j][d];
                    }
                }
            }
        }

        // Atualiza centróides
        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }

        if (changes == 0) {
            break;
        }

        iterations++;
    }
}

int main() {
    double (*points)[NUM_DIMENSIONS] = malloc(NUM_POINTS * sizeof(*points));
    int *labels = malloc(NUM_POINTS * sizeof(*labels));
    double centroids[K][NUM_DIMENSIONS];

    // Inicializa pontos e centróides com valores aleatórios
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
            points[i][d] = rand() % 100;
        }
        labels[i] = 0;
    }

    for (int j = 0; j < K; j++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
            centroids[j][d] = rand() % 100;
        }
    }

    double start = omp_get_wtime();
    kmeans_parallel(points, labels, centroids);
    double end = omp_get_wtime();

    printf("Tempo de execução: %f segundos\n", end - start);

    // Libera a memória
    free(points);
    free(labels);

    return 0;
}
