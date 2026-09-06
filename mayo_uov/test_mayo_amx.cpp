/*
 * Correctness and Performance Benchmark for MAYO / UOV AMX Acceleration
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>

extern "C" {
#include "gf16.h"
#include "mayo_amx.h"
#include "m1cycles.h"
#include "aarch64.h"
#include "amx.h"
}

class MAYOAMXTest : public ::testing::Test {};

/* Test GF(16) addition and multiplication properties */
TEST_F(MAYOAMXTest, GF16Properties) {
    /* Test commutativity: a * b = b * a */
    for (uint8_t a = 0; a < 16; a++) {
        for (uint8_t b = 0; b < 16; b++) {
            EXPECT_EQ(gf16_mul(a, b), gf16_mul(b, a));
        }
    }

    /* Test identity: a * 1 = a */
    for (uint8_t a = 0; a < 16; a++) {
        EXPECT_EQ(gf16_mul(a, 1), a);
        EXPECT_EQ(gf16_mul(a, 0), 0);
    }
}

/* Test quadratic form evaluation match between reference and AMX */
TEST_F(MAYOAMXTest, QuadFormMatchMAYO1) {
    int m = 64;  /* MAYO-1 parameters */
    int n = 66;

    uint8_t *P = (uint8_t *)malloc(m * n * n);
    uint8_t *x = (uint8_t *)malloc(n);
    uint8_t *y_ref = (uint8_t *)malloc(m);
    uint8_t *y_amx = (uint8_t *)malloc(m);

    srand(12345);
    for (int i = 0; i < m * n * n; i++) P[i] = rand() % 16;
    for (int i = 0; i < n; i++) x[i] = rand() % 16;

    /* Compute reference */
    gf16_eval_quad_forms_ref(y_ref, P, x, m, n);

    /* Compute AMX */
    amx_mayo_eval_quad_forms(y_amx, P, x, m, n);

    for (int k = 0; k < m; k++) {
        ASSERT_EQ(y_amx[k], y_ref[k]) << "Mismatch in quadratic form evaluation at k=" << k;
    }

    free(P);
    free(x);
    free(y_ref);
    free(y_amx);
}

/* Comparative Benchmark: C Reference vs. AMX Acceleration */
TEST_F(MAYOAMXTest, CompareRefVsAMX) {
    int m = 64;  /* MAYO-1 parameters */
    int n = 66;

    uint8_t *P = (uint8_t *)malloc(m * n * n);
    uint8_t *x = (uint8_t *)malloc(n);
    uint8_t *y_ref = (uint8_t *)malloc(m);
    uint8_t *y_amx = (uint8_t *)malloc(m);

    srand(2026);
    for (int i = 0; i < m * n * n; i++) P[i] = rand() % 16;
    for (int i = 0; i < n; i++) x[i] = rand() % 16;

    const int trials = 200;

    /* Time Reference C Implementation */
    auto t0_ref = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < trials; iter++) {
        gf16_eval_quad_forms_ref(y_ref, P, x, m, n);
    }
    auto t1_ref = std::chrono::high_resolution_clock::now();
    double time_ref_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_ref - t0_ref).count() / (double)trials / 1000.0;

    /* Time AMX Implementation */
    auto t0_amx = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < trials; iter++) {
        amx_mayo_eval_quad_forms(y_amx, P, x, m, n);
    }
    auto t1_amx = std::chrono::high_resolution_clock::now();
    double time_amx_us = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1_amx - t0_amx).count() / (double)trials / 1000.0;

    double speedup = time_ref_us / time_amx_us;

    printf("\n========================================================================\n");
    printf("MAYO-1 MULTIVARIATE QUADRATIC FORM EVALUATION BENCHMARK (Apple M3)\n");
    printf("Parameters: m = %d quadratic forms, n = %d variables over GF(16)\n", m, n);
    printf("------------------------------------------------------------------------\n");
    printf("  Reference C Implementation : %8.3f us\n", time_ref_us);
    printf("  AMX Acceleration (Ours)   : %8.3f us\n", time_amx_us);
    printf("  Speedup Factor            : %8.2fx\n", speedup);
    printf("========================================================================\n");
    fflush(stdout);

    free(P);
    free(x);
    free(y_ref);
    free(y_amx);
}
