#include <gtest/gtest.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_amx.h"
#include "amx_vector_search.h"

class AMXVectorSearchTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AMXVectorSearchTest, CorrectnessCheckAllDims) {
    const int dims[] = {128, 256, 512, 768, 1024, 1536};
    const int num_dims = sizeof(dims) / sizeof(dims[0]);
    const int B = 8;
    const int N = 128;

    for (int idx = 0; idx < num_dims; idx++) {
        int d = dims[idx];

        int8_t *Q = (int8_t *)malloc((size_t)B * d);
        int8_t *D = (int8_t *)malloc((size_t)N * d);
        int32_t *ref_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));
        int32_t *neon_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));
        int32_t *amx_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));

        ASSERT_NE(Q, nullptr);
        ASSERT_NE(D, nullptr);

        /* Fill Q and D with pseudo-random test values */
        for (int i = 0; i < B * d; i++) {
            Q[i] = (int8_t)((rand() % 255) - 128);
        }
        for (int i = 0; i < N * d; i++) {
            D[i] = (int8_t)((rand() % 255) - 128);
        }

        /* Compute scores using C Scalar Reference */
        ref_int8_vector_search(ref_scores, Q, D, B, N, d);

        /* Compute scores using ARM NEON SIMD */
        neon_int8_vector_search(neon_scores, Q, D, B, N, d);

        /* Compute scores using Native Apple AMX */
        amx_int8_vector_search(amx_scores, Q, D, B, N, d);

        /* Verify exact equivalence */
        for (int b = 0; b < B; b++) {
            for (int i = 0; i < N; i++) {
                int index = b * N + i;
                EXPECT_EQ(ref_scores[index], neon_scores[index])
                    << "Mismatch between REF and NEON at dim=" << d << ", b=" << b << ", i=" << i;
                EXPECT_EQ(ref_scores[index], amx_scores[index])
                    << "Mismatch between REF and AMX at dim=" << d << ", b=" << b << ", i=" << i;
            }
        }

        free(Q);
        free(D);
        free(ref_scores);
        free(neon_scores);
        free(amx_scores);
    }
}

TEST_F(AMXVectorSearchTest, TopKRetrievalTest) {
    const int B = 4;
    const int N = 100;
    const int d = 512;
    const int K = 5;

    int8_t *Q = (int8_t *)malloc((size_t)B * d);
    int8_t *D = (int8_t *)malloc((size_t)N * d);
    int32_t *scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));
    int32_t *top_indices = (int32_t *)malloc((size_t)B * K * sizeof(int32_t));
    int32_t *top_scores = (int32_t *)malloc((size_t)B * K * sizeof(int32_t));

    for (int i = 0; i < B * d; i++) Q[i] = (int8_t)((rand() % 255) - 128);
    for (int i = 0; i < N * d; i++) D[i] = (int8_t)((rand() % 255) - 128);

    amx_int8_vector_search(scores, Q, D, B, N, d);
    amx_find_top_k(top_indices, top_scores, scores, B, N, K);

    /* Verify top-K scores are sorted in descending order */
    for (int b = 0; b < B; b++) {
        for (int k = 0; k < K - 1; k++) {
            EXPECT_GE(top_scores[b * K + k], top_scores[b * K + k + 1]);
        }
    }

    free(Q);
    free(D);
    free(scores);
    free(top_indices);
    free(top_scores);
}
