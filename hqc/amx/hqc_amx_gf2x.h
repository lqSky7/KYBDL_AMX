#ifndef HQC_AMX_GF2X_H
#define HQC_AMX_GF2X_H

#include <stdint.h>
#include <stddef.h>
#include "params.h"
#include "gf2x.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Dense × Dense polynomial multiplication (AMX integer schoolbook)
 */
void gf2x_mul_amx(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n);

/*
 * Sparse × Dense polynomial multiplication (AMX vectorized accumulate)
 */
void gf2x_mul_sparse_amx(uint64_t *c, const uint32_t *positions, size_t weight,
                           const uint64_t *b, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* HQC_AMX_GF2X_H */
