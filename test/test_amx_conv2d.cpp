#include <gtest/gtest.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_amx.h"
#include "amx_conv2d.h"

class AMXConv2DTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AMXConv2DTest, CorrectnessCheck3x3Kernel) {
    const int resolutions[][2] = {{32, 32}, {64, 64}, {128, 128}};
    const int num_res = sizeof(resolutions) / sizeof(resolutions[0]);
    const int C_in = 3;
    const int C_out = 32;
    const int Kh = 3, Kw = 3;
    const int pad_h = 1, pad_w = 1;
    const int stride_h = 1, stride_w = 1;

    for (int r = 0; r < num_res; r++) {
        int H = resolutions[r][0];
        int W = resolutions[r][1];

        int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
        int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
        int N = H_out * W_out;

        int8_t *input = (int8_t *)malloc((size_t)C_in * H * W);
        int8_t *weight = (int8_t *)malloc((size_t)C_out * C_in * Kh * Kw);
        int32_t *ref_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
        int32_t *neon_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
        int32_t *amx_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));

        ASSERT_NE(input, nullptr);
        ASSERT_NE(weight, nullptr);
        ASSERT_NE(ref_out, nullptr);
        ASSERT_NE(neon_out, nullptr);
        ASSERT_NE(amx_out, nullptr);

        for (int i = 0; i < C_in * H * W; i++) input[i] = (int8_t)((rand() % 255) - 128);
        for (int i = 0; i < C_out * C_in * Kh * Kw; i++) weight[i] = (int8_t)((rand() % 255) - 128);

        ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
        neon_int8_conv2d(neon_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
        amx_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C_out; c++) {
                int idx = n * C_out + c;
                EXPECT_EQ(ref_out[idx], neon_out[idx])
                    << "Mismatch REF vs NEON (3x3) at res=" << H << "x" << W << ", pixel=" << n << ", channel=" << c;
                EXPECT_EQ(ref_out[idx], amx_out[idx])
                    << "Mismatch REF vs AMX (3x3) at res=" << H << "x" << W << ", pixel=" << n << ", channel=" << c;
            }
        }

        free(input);
        free(weight);
        free(ref_out);
        free(neon_out);
        free(amx_out);
    }
}

TEST_F(AMXConv2DTest, CorrectnessCheck5x5Kernel) {
    const int resolutions[][2] = {{32, 32}, {64, 64}, {128, 128}};
    const int num_res = sizeof(resolutions) / sizeof(resolutions[0]);
    const int C_in = 3;
    const int C_out = 32;
    const int Kh = 5, Kw = 5;
    const int pad_h = 2, pad_w = 2;
    const int stride_h = 1, stride_w = 1;

    for (int r = 0; r < num_res; r++) {
        int H = resolutions[r][0];
        int W = resolutions[r][1];

        int H_out = (H + 2 * pad_h - Kh) / stride_h + 1;
        int W_out = (W + 2 * pad_w - Kw) / stride_w + 1;
        int N = H_out * W_out;

        int8_t *input = (int8_t *)malloc((size_t)C_in * H * W);
        int8_t *weight = (int8_t *)malloc((size_t)C_out * C_in * Kh * Kw);
        int32_t *ref_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
        int32_t *neon_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));
        int32_t *amx_out = (int32_t *)malloc((size_t)N * C_out * sizeof(int32_t));

        ASSERT_NE(input, nullptr);
        ASSERT_NE(weight, nullptr);
        ASSERT_NE(ref_out, nullptr);
        ASSERT_NE(neon_out, nullptr);
        ASSERT_NE(amx_out, nullptr);

        for (int i = 0; i < C_in * H * W; i++) input[i] = (int8_t)((rand() % 255) - 128);
        for (int i = 0; i < C_out * C_in * Kh * Kw; i++) weight[i] = (int8_t)((rand() % 255) - 128);

        ref_int8_conv2d(ref_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
        neon_int8_conv2d(neon_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);
        amx_int8_conv2d(amx_out, input, weight, C_in, H, W, C_out, Kh, Kw, pad_h, pad_w, stride_h, stride_w);

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C_out; c++) {
                int idx = n * C_out + c;
                EXPECT_EQ(ref_out[idx], neon_out[idx])
                    << "Mismatch REF vs NEON (5x5) at res=" << H << "x" << W << ", pixel=" << n << ", channel=" << c;
                EXPECT_EQ(ref_out[idx], amx_out[idx])
                    << "Mismatch REF vs AMX (5x5) at res=" << H << "x" << W << ", pixel=" << n << ", channel=" << c;
            }
        }

        free(input);
        free(weight);
        free(ref_out);
        free(neon_out);
        free(amx_out);
    }
}
