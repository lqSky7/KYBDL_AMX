#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h>
#include "aarch64.h"
#include "amx.h"
#include "amx_conv2d.h"

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    printf("=========================================================\n");
    printf(" AMX & NEON 5x5 KERNEL BOTTLENECK DIAGNOSTIC ANALYSIS\n");
    printf("=========================================================\n");
    fflush(stdout);

    int H = 64, W = 64, C_in = 3, C_out = 32;
    int Kh = 5, Kw = 5, pad_h = 2, pad_w = 2, stride_h = 1, stride_w = 1;
    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw; /* 75 */

    int8_t *input = (int8_t *)malloc((size_t)C_in * H * W);
    int8_t *weight = (int8_t *)malloc((size_t)C_out * C_in * Kh * Kw);
    int32_t *ref_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
    int32_t *neon_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
    int32_t *amx_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));

    for (int i = 0; i < C_in * H * W; i++) input[i] = (int8_t)((rand() % 255) - 128);
    for (int i = 0; i < C_out * C_in * Kh * Kw; i++) weight[i] = (int8_t)((rand() % 255) - 128);

    /* Test 1: Scalar C */
    double t0 = get_time_sec();
    for (int t = 0; t < 20; t++) {
        ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
    }
    double t_ref = (get_time_sec() - t0) / 20.0 * 1000.0;

    /* Test 2: Manual NEON */
    t0 = get_time_sec();
    for (int t = 0; t < 20; t++) {
        neon_int8_conv2d(neon_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
    }
    double t_neon = (get_time_sec() - t0) / 20.0 * 1000.0;

    /* Test 3: AMX Per-Pixel Launch & Drain */
    t0 = get_time_sec();
    for (int t = 0; t < 20; t++) {
        amx_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
    }
    double t_amx = (get_time_sec() - t0) / 20.0 * 1000.0;

    printf("Kernel Config: 64x64 Image, 5x5 Kernel (K=75), Cin=3, Cout=32\n");
    printf("Total Output Dots (N x Cout): %d x %d = %d\n", N, C_out, N * C_out);
    printf("Scalar C Latency:  %8.4f ms\n", t_ref);
    printf("Manual NEON:       %8.4f ms (Speedup vs C: %.2fx)\n", t_neon, t_ref / t_neon);
    printf("AMX Per-Element:   %8.4f ms (Slowdown vs C: %.2fx)\n", t_amx, t_amx / t_ref);
    fflush(stdout);

    free(input); free(weight); free(ref_out); free(neon_out); free(amx_out);
    return 0;
}
