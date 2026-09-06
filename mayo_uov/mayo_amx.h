/*
 * AMX Acceleration Engine for MAYO / UOV Post-Quantum Signatures
 *
 * Accelerates multivariate quadratic evaluations y_k = x^T * P_k * x over GF(16)
 * using Apple's AMX coprocessor (MAC8, GENLUT, Z register accumulation).
 *
 * Parameters (matching MAYO-1 / UOV specs):
 * - MAYO_1:  n = 66, m = 64, field = GF(16)
 * - MAYO_2:  n = 78, m = 64, field = GF(16)
 * - MAYO_3:  n = 99, m = 96, field = GF(16)
 * - MAYO_5:  n = 133, m = 128, field = GF(16)
 *
 * License: CC0 1.0 Universal / MIT
 */

#ifndef MAYO_AMX_H
#define MAYO_AMX_H

#include <stdint.h>
#include <stddef.h>
#include "gf16.h"

#define MAYO_N 66
#define MAYO_M 64

/* AMX-accelerated GF(16) matrix-vector product: c = A * b */
void amx_gf16_matvec_mul(uint8_t *c, const uint8_t *A, const uint8_t *b, int rows, int cols);

/* AMX-accelerated batch quadratic form evaluation for MAYO / UOV */
void amx_mayo_eval_quad_forms(uint8_t *y, const uint8_t *P, const uint8_t *x, int m, int n);

#endif /* MAYO_AMX_H */
