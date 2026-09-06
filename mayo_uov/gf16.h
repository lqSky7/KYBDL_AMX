/*
 * GF(16) Finite Field Arithmetic for MAYO / UOV Post-Quantum Signatures
 *
 * GF(16) = GF(2)[x] / (x^4 + x + 1)
 * Field elements are represented as 4-bit integers (nibbles in [0, 15]).
 *
 * License: CC0 1.0 Universal / MIT
 */

#ifndef GF16_H
#define GF16_H

#include <stdint.h>
#include <stddef.h>

#define GF16_ORDER 16

/* Addition in GF(16) is bitwise XOR */
static inline uint8_t gf16_add(uint8_t a, uint8_t b) {
    return (a ^ b) & 0x0F;
}

/* Multiplication lookup table for GF(16) mod (x^4 + x + 1) */
extern const uint8_t GF16_MUL_TABLE[16][16];

static inline uint8_t gf16_mul(uint8_t a, uint8_t b) {
    return GF16_MUL_TABLE[a & 0x0F][b & 0x0F];
}

/* Vectorized matrix-vector multiplication over GF(16) */
void gf16_matvec_mul_ref(uint8_t *c, const uint8_t *A, const uint8_t *b, int rows, int cols);

/* Vectorized quadratic form evaluation: y_i = x^T * P_i * x over GF(16) */
void gf16_eval_quad_forms_ref(uint8_t *y, const uint8_t *P, const uint8_t *x, int m, int n);

#endif /* GF16_H */
