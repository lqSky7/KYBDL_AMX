#ifndef SNTRUP_AMX_POLYMUL_H
#define SNTRUP_AMX_POLYMUL_H

#include <stdint.h>
#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AMX polynomial multiplication: c = a * b mod (x^p - x - 1) mod q
 * Caller must have called AMX_SET() beforehand. */
void poly_mul_amx(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                  const int16_t b[SNTRUP_P]);

#ifdef __cplusplus
}
#endif

#endif /* SNTRUP_AMX_POLYMUL_H */
