#include "amx_conv2d.h"
#include "aarch64.h"
#include "amx.h"

#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((constructor)) void init_amx_conv2d(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    AMX_SET();
}

/* =========================================================================
 * 1. im2col Transformation
 * ========================================================================= */
void im2col_int8(int8_t *data_col,
                 const int8_t *data_im,
                 int C_in, int H, int W,
                 int Kh, int Kw,
                 int pad_h, int pad_w,
                 int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int K = C_in * Kh * Kw;

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
                            col_row[k_idx] = data_im[(c * H + h_in) * W + w_in];
                        } else {
                            col_row[k_idx] = 0; /* Padding zero */
                        }
                        k_idx++;
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * 2. Scalar C Reference 2D Convolution Implementation
 * ========================================================================= */
void ref_int8_conv2d(int32_t *output,
                     const int8_t *input,
                     const int8_t *weight,
                     int C_in, int H, int W,
                     int C_out, int Kh, int Kw,
                     int pad_h, int pad_w,
                     int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw;

    int8_t *data_col = (int8_t *)malloc((size_t)N * K);
    if (!data_col) return;

    im2col_int8(data_col, input, C_in, H, W, Kh, Kw, pad_h, pad_w, stride_h, stride_w);

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

/* =========================================================================
 * 3. High-Performance ARM NEON SIMD 2D Convolution Engine
 * ========================================================================= */
void neon_int8_conv2d(int32_t *output,
                      const int8_t *input,
                      const int8_t *weight,
                      int C_in, int H, int W,
                      int C_out, int Kh, int Kw,
                      int pad_h, int pad_w,
                      int stride_h, int stride_w) {
    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw;

    int8_t *data_col = (int8_t *)malloc((size_t)N * K);
    if (!data_col) return;

    im2col_int8(data_col, input, C_in, H, W, Kh, Kw, pad_h, pad_w, stride_h, stride_w);

    for (int n = 0; n < N; n++) {
        const int8_t *col_row = &data_col[n * K];
        int32_t *out_row = &output[n * C_out];

        for (int c = 0; c < C_out; c++) {
            const int8_t *w_row = &weight[c * K];
            int32x4_t vsum0 = vdupq_n_s32(0);
            int32x4_t vsum1 = vdupq_n_s32(0);

            int k = 0;
            for (; k <= K - 32; k += 32) {
                int8x16_t c0 = vld1q_s8(&col_row[k]);
                int8x16_t w0 = vld1q_s8(&w_row[k]);
                int8x16_t c1 = vld1q_s8(&col_row[k + 16]);
                int8x16_t w1 = vld1q_s8(&w_row[k + 16]);

#if defined(__ARM_FEATURE_DOTPROD)
                vsum0 = vdotq_s32(vsum0, c0, w0);
                vsum1 = vdotq_s32(vsum1, c1, w1);
#else
                int16x8_t p0_l = vmull_s8(vget_low_s8(c0), vget_low_s8(w0));
                int16x8_t p0_h = vmull_high_s8(c0, w0);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(p0_l));
                vsum0 = vaddw_high_s16(vsum0, p0_l);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(p0_h));
                vsum0 = vaddw_high_s16(vsum0, p0_h);

                int16x8_t p1_l = vmull_s8(vget_low_s8(c1), vget_low_s8(w1));
                int16x8_t p1_h = vmull_high_s8(c1, w1);
                vsum1 = vaddw_s16(vsum1, vget_low_s16(p1_l));
                vsum1 = vaddw_high_s16(vsum1, p1_l);
                vsum1 = vaddw_s16(vsum1, vget_low_s16(p1_h));
                vsum1 = vaddw_high_s16(vsum1, p1_h);
#endif
            }
            for (; k <= K - 16; k += 16) {
                int8x16_t c0 = vld1q_s8(&col_row[k]);
                int8x16_t w0 = vld1q_s8(&w_row[k]);
#if defined(__ARM_FEATURE_DOTPROD)
                vsum0 = vdotq_s32(vsum0, c0, w0);
#else
                int16x8_t p0_l = vmull_s8(vget_low_s8(c0), vget_low_s8(w0));
                int16x8_t p0_h = vmull_high_s8(c0, w0);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(p0_l));
                vsum0 = vaddw_high_s16(vsum0, p0_l);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(p0_h));
                vsum0 = vaddw_high_s16(vsum0, p0_h);
#endif
            }

            int32_t total = vaddvq_s32(vaddq_s32(vsum0, vsum1));

            /* Scalar tail */
            for (; k < K; k++) {
                total += (int32_t)col_row[k] * (int32_t)w_row[k];
            }

            out_row[c] = total;
        }
    }

    free(data_col);
}

/* =========================================================================
 * 4. Native Apple AMX 32x32 Tiled Hardware 2D Convolution Engine
 * ========================================================================= */
#define AMX_KBATCH 2

void amx_int8_conv2d(int32_t *output,
                     const int8_t *input,
                     const int8_t *weight,
                     int C_in, int H, int W,
                     int C_out, int Kh, int Kw,
                     int pad_h, int pad_w,
                     int stride_h, int stride_w) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
    int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
    int N = H_out * W_out;
    int K = C_in * Kh * Kw;

    /* Zero output buffer */
    memset(output, 0, (size_t)N * C_out * sizeof(int32_t));

    int num_c_tiles = (C_out + 31) / 32;
    int num_n_tiles = (N + 31) / 32;

    /* Pack filter weights into 32-channel aligned blocks of shape (C_out/32) x K x 32 int16_t */
    int16_t *w_packed;
    posix_memalign((void **)&w_packed, 128, (size_t)num_c_tiles * K * 32 * sizeof(int16_t));
    if (!w_packed) return;
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

    /* Spatial tile scratch buffer: K x 32 int16_t */
    int16_t *a_tile;
    posix_memalign((void **)&a_tile, 128, (size_t)K * 32 * sizeof(int16_t));
    if (!a_tile) {
        free(w_packed);
        return;
    }

    int16_t z_buf[32 * 32] __attribute__((aligned(128)));

    /* Outer loop over spatial patch blocks (32 pixels per block) */
    for (int nt = 0; nt < num_n_tiles; nt++) {
        int n0 = nt * 32;
        int n_block = (N - n0 < 32) ? (N - n0) : 32;

        /* Directly unpack sliding window patches for 32 pixels into a_tile */
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

        /* Loop over output channel blocks (32 channels per block) */
        for (int ct = 0; ct < num_c_tiles; ct++) {
            int c0 = ct * 32;
            int c_block = (C_out - c0 < 32) ? (C_out - c0) : 32;

            /* Accumulate over reduction dimension K in batches to prevent 16-bit accumulator overflow */
            for (int k0 = 0; k0 < K; k0 += AMX_KBATCH) {
                int k_end = (k0 + AMX_KBATCH <= K) ? (k0 + AMX_KBATCH) : K;

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

                /* Drain 32 rows of Z accumulator to local L1 cache buffer */
                for (int r = 0; r < n_block; r++) {
                    AMX_STZ(AMX_PTR(&z_buf[32 * r]) | STZ_Z_ROW(2 * r));
                }

                /* Vectorized NEON SIMD accumulation into 32-bit output */
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

    free(a_tile);
    free(w_packed);
}
