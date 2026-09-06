#include "amx_vector_search.h"
#include "aarch64.h"
#include "amx.h"

#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((constructor)) void init_vector_search_amx(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    AMX_SET();
}

/* =========================================================================
 * 1. C Scalar Reference Implementation
 * ========================================================================= */
void ref_int8_vector_search(int32_t *scores, const int8_t *Q, const int8_t *D, int B, int N, int d) {
    for (int b = 0; b < B; b++) {
        const int8_t *q_vec = &Q[b * d];
        int32_t *b_scores = &scores[b * N];

        for (int i = 0; i < N; i++) {
            const int8_t *d_vec = &D[i * d];
            int32_t acc = 0;
            for (int k = 0; k < d; k++) {
                acc += (int32_t)q_vec[k] * (int32_t)d_vec[k];
            }
            b_scores[i] = acc;
        }
    }
}

/* =========================================================================
 * 2. High-Performance ARM NEON SIMD Matrix Search Engine
 * ========================================================================= */
void neon_int8_vector_search(int32_t *scores, const int8_t *Q, const int8_t *D, int B, int N, int d) {
    for (int b = 0; b < B; b++) {
        const int8_t *q_vec = &Q[b * d];
        int32_t *b_scores = &scores[b * N];

        for (int i = 0; i < N; i++) {
            const int8_t *d_vec = &D[i * d];
            int32x4_t vsum0 = vdupq_n_s32(0);
            int32x4_t vsum1 = vdupq_n_s32(0);

            int k = 0;
            for (; k <= d - 32; k += 32) {
                int8x16_t q0 = vld1q_s8(&q_vec[k]);
                int8x16_t d0 = vld1q_s8(&d_vec[k]);
                int8x16_t q1 = vld1q_s8(&q_vec[k + 16]);
                int8x16_t d1 = vld1q_s8(&d_vec[k + 16]);

#if defined(__ARM_FEATURE_DOTPROD)
                vsum0 = vdotq_s32(vsum0, q0, d0);
                vsum1 = vdotq_s32(vsum1, q1, d1);
#else
                int16x8_t prod0_low = vmull_s8(vget_low_s8(q0), vget_low_s8(d0));
                int16x8_t prod0_high = vmull_high_s8(q0, d0);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(prod0_low));
                vsum0 = vaddw_high_s16(vsum0, prod0_low);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(prod0_high));
                vsum0 = vaddw_high_s16(vsum0, prod0_high);

                int16x8_t prod1_low = vmull_s8(vget_low_s8(q1), vget_low_s8(d1));
                int16x8_t prod1_high = vmull_high_s8(q1, d1);
                vsum1 = vaddw_s16(vsum1, vget_low_s16(prod1_low));
                vsum1 = vaddw_high_s16(vsum1, prod1_low);
                vsum1 = vaddw_s16(vsum1, vget_low_s16(prod1_high));
                vsum1 = vaddw_high_s16(vsum1, prod1_high);
#endif
            }
            for (; k <= d - 16; k += 16) {
                int8x16_t q0 = vld1q_s8(&q_vec[k]);
                int8x16_t d0 = vld1q_s8(&d_vec[k]);
#if defined(__ARM_FEATURE_DOTPROD)
                vsum0 = vdotq_s32(vsum0, q0, d0);
#else
                int16x8_t prod0_low = vmull_s8(vget_low_s8(q0), vget_low_s8(d0));
                int16x8_t prod0_high = vmull_high_s8(q0, d0);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(prod0_low));
                vsum0 = vaddw_high_s16(vsum0, prod0_low);
                vsum0 = vaddw_s16(vsum0, vget_low_s16(prod0_high));
                vsum0 = vaddw_high_s16(vsum0, prod0_high);
#endif
            }

            int32_t total = vaddvq_s32(vaddq_s32(vsum0, vsum1));

            /* Scalar tail */
            for (; k < d; k++) {
                total += (int32_t)q_vec[k] * (int32_t)d_vec[k];
            }

            b_scores[i] = total;
        }
    }
}

/* =========================================================================
 * 3. Direct Native 8-bit AMX Matrix Hardware Engine Implementation
 * ========================================================================= */
void amx_int8_vector_search(int32_t *scores, const int8_t *Q, const int8_t *D, int B, int N, int d) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    AMX_SET();

    for (int b = 0; b < B; b++) {
        const int8_t *q_vec = &Q[b * d];
        int32_t *b_scores = &scores[b * N];

        int k = 0;
        /* Direct 64-byte AMX processing */
        for (; k <= d - 64; k += 64) {
            /* Load 64 bytes of query directly into X0 */
            AMX_LDX(AMX_PTR(&q_vec[k]) | LDX_REG(0));

            for (int i = 0; i < N; i++) {
                const int8_t *d_vec = &D[i * d];

                /* Load 64 bytes of database vector directly into Y0 */
                AMX_LDY(AMX_PTR(&d_vec[k]) | LDY_REG(0));

                /* Direct 8-bit integer outer product into Z row 0 */
                AMX_MATINT(MATINT_ALU_MODE_Z_ADD_X_MUL_Y | MATINT_X_REG(0) | MATINT_Y_REG(0) | MATINT_Z_ROW(0));

                /* Store Z row accumulators directly */
                int32_t z_out[16];
                AMX_STZ(AMX_PTR(z_out) | STZ_Z_ROW(0));

                int32x4_t vsum = vld1q_s32(&z_out[0]);
                vsum = vaddq_s32(vsum, vld1q_s32(&z_out[4]));
                vsum = vaddq_s32(vsum, vld1q_s32(&z_out[8]));
                vsum = vaddq_s32(vsum, vld1q_s32(&z_out[12]));

                b_scores[i] += vaddvq_s32(vsum);
            }
        }

        /* Tail processing for remaining elements */
        if (k < d) {
            for (int i = 0; i < N; i++) {
                const int8_t *d_vec = &D[i * d];
                int32_t tail_acc = 0;
                for (int rem = k; rem < d; rem++) {
                    tail_acc += (int32_t)q_vec[rem] * (int32_t)d_vec[rem];
                }
                b_scores[i] += tail_acc;
            }
        }
    }

    AMX_CLR();
}

/* =========================================================================
 * 4. Cosine Similarity Engine
 * ========================================================================= */
void amx_cosine_similarity_search(float *cosine_scores, const int8_t *Q, const int8_t *D,
                                  const float *norm_Q, const float *norm_D, int B, int N, int d) {
    int32_t *raw_scores = (int32_t *)malloc((size_t)B * N * sizeof(int32_t));
    if (!raw_scores) return;

    /* Perform AMX matrix dot products */
    amx_int8_vector_search(raw_scores, Q, D, B, N, d);

    /* Normalize to [-1.0, 1.0] cosine similarity range */
    for (int b = 0; b < B; b++) {
        float n_q = norm_Q[b];
        if (n_q == 0.0f) n_q = 1.0f;

        for (int i = 0; i < N; i++) {
            float n_d = norm_D[i];
            if (n_d == 0.0f) n_d = 1.0f;

            float dot = (float)raw_scores[b * N + i];
            cosine_scores[b * N + i] = dot / (n_q * n_d);
        }
    }

    free(raw_scores);
}

/* =========================================================================
 * 5. Top-K Index Helper
 * ========================================================================= */
typedef struct {
    int32_t score;
    int32_t index;
} ScoreIdxPair;

static int compare_pairs(const void *a, const void *b) {
    const ScoreIdxPair *p1 = (const ScoreIdxPair *)a;
    const ScoreIdxPair *p2 = (const ScoreIdxPair *)b;
    if (p2->score > p1->score) return 1;
    if (p2->score < p1->score) return -1;
    return 0;
}

void amx_find_top_k(int32_t *top_indices, int32_t *top_scores, const int32_t *scores, int B, int N, int K) {
    ScoreIdxPair *pairs = (ScoreIdxPair *)malloc((size_t)N * sizeof(ScoreIdxPair));
    if (!pairs) return;

    for (int b = 0; b < B; b++) {
        const int32_t *b_scores = &scores[b * N];
        for (int i = 0; i < N; i++) {
            pairs[i].score = b_scores[i];
            pairs[i].index = i;
        }

        qsort(pairs, (size_t)N, sizeof(ScoreIdxPair), compare_pairs);

        for (int k = 0; k < K && k < N; k++) {
            top_scores[b * K + k] = pairs[k].score;
            top_indices[b * K + k] = pairs[k].index;
        }
    }

    free(pairs);
}
