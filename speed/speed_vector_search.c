#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "speed.h"
#include "amx_vector_search.h"

#define NTESTS 50

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("======================================================================\n");
    printf("  APPLE AMX QUANTIZED VECTOR SIMILARITY SEARCH BENCHMARK SUITE       \n");
    printf("======================================================================\n\n");
    fflush(stdout);

    const int dimensions[] = {128, 512, 1024, 1536};
    const int num_dims = sizeof(dimensions) / sizeof(dimensions[0]);
    const int B = 32;       /* Batch of 32 queries */
    const int N = 2000;     /* 2,000 database vectors */

    for (int idx = 0; idx < num_dims; idx++) {
        int d = dimensions[idx];

        int8_t *Q = (int8_t *)malloc((size_t)B * d);
        int8_t *D = (int8_t *)malloc((size_t)N * d);
        int32_t *ref_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));
        int32_t *neon_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));
        int32_t *amx_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));

        if (!Q || !D || !ref_scores || !neon_scores || !amx_scores) {
            fprintf(stderr, "Allocation failed!\n");
            return 1;
        }

        for (int i = 0; i < B * d; i++) Q[i] = (int8_t)((rand() % 255) - 128);
        for (int i = 0; i < N * d; i++) D[i] = (int8_t)((rand() % 255) - 128);

        /* 1. Benchmark C Scalar Reference */
        double t0 = get_time_sec();
        for (int t = 0; t < NTESTS; t++) {
            ref_int8_vector_search(ref_scores, Q, D, B, N, d);
            DoNotOptimize(ref_scores[0]);
        }
        double t_ref = (get_time_sec() - t0) / NTESTS;

        /* 2. Benchmark ARM NEON SIMD */
        t0 = get_time_sec();
        for (int t = 0; t < NTESTS; t++) {
            neon_int8_vector_search(neon_scores, Q, D, B, N, d);
            DoNotOptimize(neon_scores[0]);
        }
        double t_neon = (get_time_sec() - t0) / NTESTS;

        /* 3. Benchmark Native Apple AMX Hardware Engine */
        t0 = get_time_sec();
        for (int t = 0; t < NTESTS; t++) {
            amx_int8_vector_search(amx_scores, Q, D, B, N, d);
            DoNotOptimize(amx_scores[0]);
        }
        double t_amx = (get_time_sec() - t0) / NTESTS;

        double ops = 2.0 * (double)B * (double)N * (double)d;
        double gops_ref = (ops / t_ref) * 1e-9;
        double gops_neon = (ops / t_neon) * 1e-9;
        double gops_amx = (ops / t_amx) * 1e-9;

        double speedup_neon_vs_ref = t_ref / t_neon;
        double speedup_amx_vs_ref = t_ref / t_amx;
        double speedup_amx_vs_neon = t_neon / t_amx;

        printf("----------------------------------------------------------------------\n");
        printf(" Embedding Dim: d = %-4d | Query Batch B = %-2d | DB Vectors N = %-4d\n", d, B, N);
        printf("----------------------------------------------------------------------\n");
        printf(" [Scalar C]     Time: %8.4f ms | Throughput: %7.2f GOPS\n", t_ref * 1000.0, gops_ref);
        printf(" [ARM NEON]     Time: %8.4f ms | Throughput: %7.2f GOPS | Speedup vs Ref:  %5.2fx\n", t_neon * 1000.0, gops_neon, speedup_neon_vs_ref);
        printf(" [Apple AMX]    Time: %8.4f ms | Throughput: %7.2f GOPS | Speedup vs Ref:  %5.2fx | Speedup vs NEON: %5.2fx\n", t_amx * 1000.0, gops_amx, speedup_amx_vs_ref, speedup_amx_vs_neon);
        printf("\n");
        fflush(stdout);

        free(Q);
        free(D);
        free(ref_scores);
        free(neon_scores);
        free(amx_scores);
    }

    printf("======================================================================\n");
    printf("  BENCHMARK COMPLETE                                                  \n");
    printf("======================================================================\n");
    fflush(stdout);

    return 0;
}
