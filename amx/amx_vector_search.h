#ifndef AMX_VECTOR_SEARCH_H
#define AMX_VECTOR_SEARCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * C Scalar Reference implementation of INT8 Vector Dot-Product Similarity Search.
 * Computes scores matrix S[B x N] where S[b, i] = sum_{k=0}^{d-1} Q[b, k] * D[i, k]
 *
 * @param scores Output score matrix of dimension [B x N] (row-major: scores[b * N + i])
 * @param Q Query matrix of dimension [B x d] (8-bit signed integer embeddings)
 * @param D Database matrix of dimension [N x d] (8-bit signed integer embeddings)
 * @param B Number of query vectors (batch size)
 * @param N Number of database vectors
 * @param d Vector embedding dimension (e.g. 128, 256, 512, 768, 1024, 1536)
 */
void ref_int8_vector_search(int32_t *scores, const int8_t *Q, const int8_t *D, int B, int N, int d);

/**
 * ARM NEON SIMD Vectorized implementation of INT8 Vector Similarity Search.
 * Leverages ARMv8 NEON 128-bit vector dot product instructions (`vdotq_s32` / `vmlal_s8`).
 */
void neon_int8_vector_search(int32_t *scores, const int8_t *Q, const int8_t *D, int B, int N, int d);

/**
 * Native Apple AMX Coprocessor Hardware Accelerated INT8 Vector Similarity Search Engine.
 * Operates in 32x32 block grid tiles, leveraging AMX 64-byte X/Y registers and 1024 8-bit MACs/cycle.
 */
void amx_int8_vector_search(int32_t *scores, const int8_t *Q, const int8_t *D, int B, int N, int d);

/**
 * Native Apple AMX Accelerated Cosine Similarity Search Engine.
 * Computes Normalized Cosine Similarity Scores: C[b, i] = S[b, i] / (norm_Q[b] * norm_D[i])
 *
 * @param cosine_scores Output float array of dimension [B x N] containing values in [-1.0, 1.0]
 */
void amx_cosine_similarity_search(float *cosine_scores, const int8_t *Q, const int8_t *D,
                                  const float *norm_Q, const float *norm_D, int B, int N, int d);

/**
 * Top-K Similarity Search helper.
 * Extracts the indices of top K highest similarity scores for each query vector in batch B.
 *
 * @param top_indices Output matrix of shape [B x K] containing database vector indices
 * @param top_scores Output matrix of shape [B x K] containing top-K similarity scores
 * @param scores Input score matrix [B x N]
 * @param B Batch size
 * @param N Database size
 * @param K Number of top results to retrieve
 */
void amx_find_top_k(int32_t *top_indices, int32_t *top_scores, const int32_t *scores, int B, int N, int K);

#ifdef __cplusplus
}
#endif

#endif // AMX_VECTOR_SEARCH_H
