#ifndef AMX_CONV2D_H
#define AMX_CONV2D_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform im2col transformation: converts a 2D image (H x W x Cin)
 * with sliding kernel (Kh x Kw) into a 2D patch matrix of size:
 * (H_out * W_out) rows x (Cin * Kh * Kw) columns.
 */
void im2col_int8(int8_t *data_col,
                 const int8_t *data_im,
                 int C_in, int H, int W,
                 int Kh, int Kw,
                 int pad_h, int pad_w,
                 int stride_h, int stride_w);

/**
 * Scalar C Reference 2D Convolution Implementation.
 * Computes Output = Weight * im2col(Input)
 * Output dimensions: (H_out * W_out) x C_out
 */
void ref_int8_conv2d(int32_t *output,
                     const int8_t *input,
                     const int8_t *weight,
                     int C_in, int H, int W,
                     int C_out, int Kh, int Kw,
                     int pad_h, int pad_w,
                     int stride_h, int stride_w);

/**
 * Optimized ARM NEON SIMD 2D Convolution Implementation.
 * Uses NEON dot-product instructions (vdotq_s32 / vmull_s8).
 */
void neon_int8_conv2d(int32_t *output,
                      const int8_t *input,
                      const int8_t *weight,
                      int C_in, int H, int W,
                      int C_out, int Kh, int Kw,
                      int pad_h, int pad_w,
                      int stride_h, int stride_w);

/**
 * Native Apple AMX Hardware-Accelerated 2D Convolution Engine.
 * Uses im2col + Tiled 32x32 AMX outer-product matrix multiplication.
 */
void amx_int8_conv2d(int32_t *output,
                     const int8_t *input,
                     const int8_t *weight,
                     int C_in, int H, int W,
                     int C_out, int Kh, int Kw,
                     int pad_h, int pad_w,
                     int stride_h, int stride_w);

#ifdef __cplusplus
}
#endif

#endif /* AMX_CONV2D_H */
