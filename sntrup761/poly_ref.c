/*
 * sntrup761 — Reference C Polynomial Multiplication (schoolbook)
 *
 * Computes c(x) = a(x) * b(x) mod (x^p - x - 1) mod q
 * using scalar schoolbook with 32-bit accumulators.
 */

#include "poly.h"
#include <string.h>

/*
 * Reduce unreduced product mod (x^p - x - 1) then mod q.
 *
 * Since x^p ≡ x + 1, each upper term c_{p+k} * x^{p+k} folds as:
 *   c_{p+k} contributes to coefficient k (via x^k) and k+1 (via x^{k+1}).
 */
void poly_reduce(int16_t c[SNTRUP_P], const int32_t prod[2 * SNTRUP_P - 1]) {
    int32_t tmp[SNTRUP_P];

    /* Copy lower p coefficients */
    memcpy(tmp, prod, SNTRUP_P * sizeof(int32_t));

    /* Fold upper coefficients: for k = 0..p-2, x^{p+k} = x^{k+1} + x^k */
    for (int k = 0; k < SNTRUP_P - 1; k++) {
        int32_t upper = prod[SNTRUP_P + k];
        tmp[k]     += upper;
        tmp[k + 1] += upper;
    }

    /* Reduce each coefficient mod q into centered range */
    for (int i = 0; i < SNTRUP_P; i++) {
        c[i] = mod_q(tmp[i]);
    }
}

/*
 * Reference schoolbook polynomial multiplication.
 * O(p^2) with 32-bit accumulators — correct but slow.
 */
void poly_mul_ref(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                  const int16_t b[SNTRUP_P]) {
    int32_t prod[2 * SNTRUP_P - 1];
    memset(prod, 0, sizeof(prod));

    for (int i = 0; i < SNTRUP_P; i++) {
        if (a[i] == 0) continue;
        for (int j = 0; j < SNTRUP_P; j++) {
            prod[i + j] += (int32_t)a[i] * (int32_t)b[j];
        }
    }

    poly_reduce(c, prod);
}
