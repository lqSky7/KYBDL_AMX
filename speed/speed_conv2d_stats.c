#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#include "speed.h"
#include "amx_conv2d.h"

#define NUM_TRIALS 30
#define WARMUP_RUNS 5

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void compute_stats(const double *times, int n, double *mean, double *sd) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += times[i];
    *mean = sum / n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = times[i] - *mean;
        var += diff * diff;
    }
    *sd = sqrt(var / (n > 1 ? (n - 1) : 1));
}

void run_kernel_benchmark(int Kh, int Kw, int pad_h, int pad_w) {
    const int resolutions[][2] = {{64, 64}, {128, 128}, {256, 256}};
    const int num_res = sizeof(resolutions) / sizeof(resolutions[0]);
    const int C_in = 3;
    const int C_out = 32;
    const int stride_h = 1, stride_w = 1;

    printf("\n======================================================================\n");
    printf(" BENCHMARK RESULTS FOR KERNEL SIZE %dx%d (30 TRIALS)\n", Kh, Kw);
    printf("======================================================================\n");

    for (int r = 0; r < num_res; r++) {
        int H = resolutions[r][0];
        int W = resolutions[r][1];

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
            return;
        }

        for (int i = 0; i < C_in * H * W; i++) input[i] = (int8_t)((rand() % 255) - 128);
        for (int i = 0; i < C_out * C_in * Kh * Kw; i++) weight[i] = (int8_t)((rand() % 255) - 128);

        double times_ref[NUM_TRIALS];
        double times_neon[NUM_TRIALS];
        double times_amx[NUM_TRIALS];

        /* Warmup */
        for (int w = 0; w < WARMUP_RUNS; w++) {
            ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            neon_int8_conv2d(neon_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            amx_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
        }

        /* 1. Benchmark C Reference */
        for (int t = 0; t < NUM_TRIALS; t++) {
            double t0 = get_time_sec();
            ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            DoNotOptimize(ref_out[0]);
            times_ref[t] = (get_time_sec() - t0) * 1000.0; /* in ms */
        }

        /* 2. Benchmark NEON */
        for (int t = 0; t < NUM_TRIALS; t++) {
            double t0 = get_time_sec();
            neon_int8_conv2d(neon_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            DoNotOptimize(neon_out[0]);
            times_neon[t] = (get_time_sec() - t0) * 1000.0; /* in ms */
        }

        /* 3. Benchmark AMX */
        for (int t = 0; t < NUM_TRIALS; t++) {
            double t0 = get_time_sec();
            amx_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
            DoNotOptimize(amx_out[0]);
            times_amx[t] = (get_time_sec() - t0) * 1000.0; /* in ms */
        }

        double mean_ref, sd_ref;
        double mean_neon, sd_neon;
        double mean_amx, sd_amx;

        compute_stats(times_ref, NUM_TRIALS, &mean_ref, &sd_ref);
        compute_stats(times_neon, NUM_TRIALS, &mean_neon, &sd_neon);
        compute_stats(times_amx, NUM_TRIALS, &mean_amx, &sd_amx);

        double speedup_vs_c = mean_ref / mean_amx;
        double speedup_vs_neon = mean_neon / mean_amx;

        double ops = 2.0 * (double)N * (double)C_out * (double)K;
        double gops_c = (ops / (mean_ref * 1e-3)) * 1e-9;
        double gops_neon = (ops / (mean_neon * 1e-3)) * 1e-9;
        double gops_amx = (ops / (mean_amx * 1e-3)) * 1e-9;

        printf("\nResolution: %dx%d | Matrix Dim (N x K): %d x %d\n", H, W, N, K);
        printf(" Scalar C: %.4f ± %.4f ms (%.2f GOPS)\n", mean_ref, sd_ref, gops_c);
        printf(" NEON:     %.4f ± %.4f ms (%.2f GOPS)\n", mean_neon, sd_neon, gops_neon);
        printf(" AMX:      %.4f ± %.4f ms (%.2f GOPS)\n", mean_amx, sd_amx, gops_amx);
        printf(" Speedup AMX vs C:    %.2fx\n", speedup_vs_c);
        printf(" Speedup AMX vs NEON: %.2fx\n", speedup_vs_neon);
        fflush(stdout);

        free(input);
        free(weight);
        free(ref_out);
        free(neon_out);
        free(amx_out);
    }
}

#include <pthread.h>
#include "aarch64.h"
#include "amx.h"

int main(void) {
    setbuf(stdout, NULL);
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    run_kernel_benchmark(3, 3, 1, 1);
    run_kernel_benchmark(5, 5, 2, 2);

    return 0;
}
