#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#include <pthread.h>
#include "aarch64.h"
#include "amx.h"

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void ref_int8_conv2d(int32_t *output, const int8_t *input, const int8_t *weight,
                     int C_in, int H, int W, int C_out, int Kh, int Kw,
                     int pad_h, int pad_w, int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw;

    int8_t *data_col = (int8_t *)malloc((size_t)N * K);
    for (int h_out = 0; h_out < H_out; h_out++) {
        for (int w_out = 0; w_out < W_out; w_out++) {
            int n = h_out * W_out + w_out;
            int8_t *col_row = &data_col[n * K];
            int k_idx = 0;
            for (int c = 0; c < C_in; c++) {
                for (int kh = 0; kh < Kh; kh++) {
                    int h_in = h_out * stride_h - pad_h + kh;
                    for (int kw = 0; kw < Kw; kw++) {
                        int w_in = w_out * stride_w - pad_w + kw;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            col_row[k_idx] = input[(c * H + h_in) * W + w_in];
                        } else {
                            col_row[k_idx] = 0;
                        }
                        k_idx++;
                    }
                }
            }
        }
    }

    for (int n = 0; n < N; n++) {
        const int8_t *col_row = &data_col[n * K];
        int32_t *out_row = &output[n * C_out];
        for (int c = 0; c < C_out; c++) {
            const int8_t *w_row = &weight[c * K];
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)col_row[k] * (int32_t)w_row[k];
            }
            out_row[c] = acc;
        }
    }
    free(data_col);
}

void amx_tiled_int8_conv2d(int32_t *output, const int8_t *input, const int8_t *weight,
                           int C_in, int H, int W, int C_out, int Kh, int Kw,
                           int pad_h, int pad_w, int stride_h, int stride_w) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    AMX_SET();

    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw;

    memset(output, 0, (size_t)N * C_out * sizeof(int32_t));

    int num_c_tiles = (C_out + 31) / 32;
    int num_n_tiles = (N + 31) / 32;

    int16_t *w_packed;
    posix_memalign((void **)&w_packed, 128, (size_t)num_c_tiles * K * 32 * sizeof(int16_t));
    memset(w_packed, 0, (size_t)num_c_tiles * K * 32 * sizeof(int16_t));

    for (int ct = 0; ct < num_c_tiles; ct++) {
        int c0 = ct * 32;
        int c_block = (C_out - c0 < 32) ? (C_out - c0) : 32;
        for (int k = 0; k < K; k++) {
            int16_t *dst = &w_packed[(ct * K + k) * 32];
            for (int col = 0; col < c_block; col++) {
                dst[col] = (int16_t)weight[(c0 + col) * K + k];
            }
        }
    }

    int16_t *a_tile;
    posix_memalign((void **)&a_tile, 128, (size_t)K * 32 * sizeof(int16_t));
    int16_t z_buf[32 * 32] __attribute__((aligned(128)));

    for (int nt = 0; nt < num_n_tiles; nt++) {
        int n0 = nt * 32;
        int n_block = (N - n0 < 32) ? (N - n0) : 32;

        memset(a_tile, 0, (size_t)K * 32 * sizeof(int16_t));

        for (int r = 0; r < n_block; r++) {
            int n = n0 + r;
            int h_out = n / W_out;
            int w_out = n % W_out;

            int k_idx = 0;
            for (int c = 0; c < C_in; c++) {
                for (int kh = 0; kh < Kh; kh++) {
                    int h_in = h_out * stride_h - pad_h + kh;
                    for (int kw = 0; kw < Kw; kw++) {
                        int w_in = w_out * stride_w - pad_w + kw;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            a_tile[k_idx * 32 + r] = (int16_t)input[(c * H + h_in) * W + w_in];
                        }
                        k_idx++;
                    }
                }
            }
        }

        for (int ct = 0; ct < num_c_tiles; ct++) {
            int c0 = ct * 32;
            int c_block = (C_out - c0 < 32) ? (C_out - c0) : 32;

            for (int k0 = 0; k0 < K; k0 += 2) {
                int k_end = (k0 + 2 <= K) ? (k0 + 2) : K;

                for (int k = k0; k < k_end; k++) {
                    const int16_t *a_ptr = &a_tile[k * 32];
                    const int16_t *w_ptr = &w_packed[(ct * K + k) * 32];

                    AMX_LDX(AMX_PTR(w_ptr) | LDX_REG(0));
                    AMX_LDY(AMX_PTR(a_ptr) | LDY_REG(0));

                    if (k == k0) {
                        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);
                    } else {
                        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0));
                    }
                }

                for (int r = 0; r < n_block; r++) {
                    AMX_STZ(AMX_PTR(&z_buf[32 * r]) | STZ_Z_ROW(2 * r));
                }

                for (int r = 0; r < n_block; r++) {
                    int32_t *out_ptr = &output[(n0 + r) * C_out + c0];
                    int16_t *z_row = &z_buf[32 * r];
                    for (int col = 0; col < c_block; col += 8) {
                        int16x8_t z_vec = vld1q_s16(&z_row[col]);
                        int32x4_t z_low = vmovl_s16(vget_low_s16(z_vec));
                        int32x4_t z_high = vmovl_high_s16(z_vec);

                        int32x4_t o_low = vld1q_s32(&out_ptr[col]);
                        int32x4_t o_high = vld1q_s32(&out_ptr[col + 4]);

                        vst1q_s32(&out_ptr[col], vaddq_s32(o_low, z_low));
                        vst1q_s32(&out_ptr[col + 4], vaddq_s32(o_high, z_high));
                    }
                }
            }
        }
    }

    AMX_CLR();
    free(a_tile);
    free(w_packed);
}

void test_config(int H, int W, int Kh, int Kw, int pad_h, int pad_w) {
    int C_in = 3, C_out = 32, stride_h = 1, stride_w = 1;
    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw;

    int8_t *input = (int8_t *)malloc((size_t)C_in * H * W);
    int8_t *weight = (int8_t *)malloc((size_t)C_out * C_in * Kh * Kw);
    int32_t *ref_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
    int32_t *amx_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));

    for (int i = 0; i < C_in * H * W; i++) input[i] = (int8_t)((rand() % 255) - 128);
    for (int i = 0; i < C_out * C_in * Kh * Kw; i++) weight[i] = (int8_t)((rand() % 255) - 128);

    ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
    amx_tiled_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);

    int errors = 0;
    for (int i = 0; i < N * C_out; i++) {
        if (ref_out[i] != amx_out[i]) errors++;
    }

    printf("Kernel %dx%d, Res %dx%d (N=%d, K=%d) -> %s (%d errors)\n",
           Kh, Kw, H, W, N, K, (errors == 0 ? "PASSED" : "FAILED"), errors);

    free(input); free(weight); free(ref_out); free(amx_out);
}

int main(void) {
    test_config(64, 64, 3, 3, 1, 1);
    test_config(128, 128, 3, 3, 1, 1);
    test_config(256, 256, 3, 3, 1, 1);
    test_config(64, 64, 5, 5, 2, 2);
    test_config(128, 128, 5, 5, 2, 2);
    test_config(256, 256, 5, 5, 2, 2);
    return 0;
}
