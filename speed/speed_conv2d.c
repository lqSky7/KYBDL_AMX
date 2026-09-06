#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "speed.h"
#include "amx_conv2d.h"

#define NTESTS 10

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("======================================================================\n");
    printf("  APPLE AMX QUANTIZED 2D IMAGE CONVOLUTION BENCHMARK SUITE          \n");
    printf("======================================================================\n\n");
    fflush(stdout);

    /* Test both shallow (RGB) and deep CNN feature map layers */
    const int configurations[][4] = {
        /* {H, W, Cin, Cout} */
        {64,  64,  3,  32},
        {128, 128, 3,  32},
        {256, 256, 3,  32},
        {64,  64,  64, 64},  /* Deep ResNet / YOLO Layer */
        {128, 128, 64, 64}   /* Deep ResNet / YOLO Layer */
    };
    const int num_configs = sizeof(configurations) / sizeof(configurations[0]);
    const int Kh = 3, Kw = 3;
    const int pad_h = 1, pad_w = 1;
    const int stride_h = 1, stride_w = 1;

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int H = configurations[cfg][0];
        int W = configurations[cfg][1];
        int C_in = configurations[cfg][2];
        int C_out = configurations[cfg][3];

        int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
        int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
        int N = H_out * W_out;
        int K = C_in * Kh * Kw;

        int8_t *input = (int8_t *)malloc((size_t)C_in * H * W);
        int8_t *weight = (int8_t *)malloc((size_t)C_out * C_in * Kh * Kw);
        int32_t *ref_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
        int32_t *neon_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
        int32_t *amx_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));

        if (!input || !weight || !ref_out || !neon_out || !amx_out) {
            fprintf(stderr, "Allocation failed!\n");
            return 1;
        }

        for (int i = 0; i < C_in * H * W; i++) input[i] = (int8_t)((rand() % 255) - 128);
        for (int i = 0; i < C_out * C_in * Kh * Kw; i++) weight[i] = (int8_t)((rand() % 255) - 128);

        /* 1. Benchmark C Scalar Reference */
        double t0 = get_time_sec();
        for (int t = 0; t < NTESTS; t++) {
            ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            DoNotOptimize(ref_out[0]);
        }
        double t_ref = (get_time_sec() - t0) / NTESTS;

        /* 2. Benchmark ARM NEON SIMD */
        t0 = get_time_sec();
        for (int t = 0; t < NTESTS; t++) {
            neon_int8_conv2d(neon_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            DoNotOptimize(neon_out[0]);
        }
        double t_neon = (get_time_sec() - t0) / NTESTS;

        /* 3. Benchmark Native Apple AMX Hardware Engine */
        t0 = get_time_sec();
        for (int t = 0; t < NTESTS; t++) {
            amx_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            DoNotOptimize(amx_out[0]);
        }
        double t_amx = (get_time_sec() - t0) / NTESTS;

        double ops = 2.0 * (double)N * (double)C_out * (double)K;
        double gops_ref = (ops / t_ref) * 1e-9;
        double gops_neon = (ops / t_neon) * 1e-9;
        double gops_amx = (ops / t_amx) * 1e-9;

        double speedup_neon_vs_ref = t_ref / t_neon;
        double speedup_amx_vs_ref = t_ref / t_amx;
        double speedup_amx_vs_neon = t_neon / t_amx;

        printf("----------------------------------------------------------------------\n");
        printf(" Layer Config: %dx%d | Cin = %d, Cout = %d | Kernel: %dx%d\n", H, W, C_in, C_out, Kh, Kw);
        printf(" Im2Col Matrix Dimensions: N (pixels) = %-6d | K (kernel) = %-4d\n", N, K);
        printf("----------------------------------------------------------------------\n");
        printf(" [Scalar C]     Time: %8.4f ms | Throughput: %7.2f GOPS\n", t_ref * 1000.0, gops_ref);
        printf(" [ARM NEON]     Time: %8.4f ms | Throughput: %7.2f GOPS | Speedup vs Ref:  %5.2fx\n", t_neon * 1000.0, gops_neon, speedup_neon_vs_ref);
        printf(" [Apple AMX]    Time: %8.4f ms | Throughput: %7.2f GOPS | Speedup vs Ref:  %5.2fx | Speedup vs NEON: %5.2fx\n", t_amx * 1000.0, gops_amx, speedup_amx_vs_ref, speedup_amx_vs_neon);
        printf("\n");
        fflush(stdout);

        free(input);
        free(weight);
        free(ref_out);
        free(neon_out);
        free(amx_out);
    }

    printf("======================================================================\n");
    printf("  BENCHMARK COMPLETE                                                  \n");
    printf("======================================================================\n");
    fflush(stdout);

    return 0;
}
