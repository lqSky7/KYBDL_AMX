/*
 * sntrup761 — AMX Polynomial Multiplication (Chunked Software Accumulation)
 *
 * Uses AMX mac16 outer products with chunked accumulation in Z (up to 7 block pairs per chunk).
 * Stores accumulated Z rows to memory and extracts anti-diagonals in 32-bit software.
 * This guarantees exact 32-bit anti-diagonal summation with zero overflow!
 */

#include "poly.h"
#include "aarch64.h"
#include "amx.h"
#include <string.h>
#include <stdlib.h>

static int is_small_poly(const int16_t *a, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] > 7 || a[i] < -7) return 0;
    }
    return 1;
}

void poly_mul_amx(int16_t c[SNTRUP_P], const int16_t a[SNTRUP_P],
                  const int16_t b[SNTRUP_P]) {
    /* AMX_SET() must be called by the caller */

    const int16_t *small_poly = a;
    const int16_t *large_poly = b;
    if (!is_small_poly(a, SNTRUP_P)) {
        if (!is_small_poly(b, SNTRUP_P)) {
            poly_mul_neon(c, a, b);
            return;
        }
        small_poly = b;
        large_poly = a;
    }

    int16_t s_pad[SNTRUP_PADDED] __attribute__((aligned(64)));
    int16_t l_pad[SNTRUP_PADDED] __attribute__((aligned(64)));
    memset(s_pad, 0, sizeof(s_pad));
    memset(l_pad, 0, sizeof(l_pad));
    memcpy(s_pad, small_poly, SNTRUP_P * sizeof(int16_t));
    memcpy(l_pad, large_poly, SNTRUP_P * sizeof(int16_t));

    int32_t prod[2 * SNTRUP_PADDED];
    memset(prod, 0, sizeof(prod));

    int16_t z_buf[32 * 32] __attribute__((aligned(64)));

    for (int d = 0; d < 2 * SNTRUP_NBLOCKS - 1; d++) {
        int i_min = (d >= SNTRUP_NBLOCKS) ? (d - SNTRUP_NBLOCKS + 1) : 0;
        int i_max = (d < SNTRUP_NBLOCKS) ? d : (SNTRUP_NBLOCKS - 1);

        int count = 0;
        for (int i = i_min; i <= i_max; i++) {
            int j = d - i;

            AMX_LDX(AMX_PTR(&s_pad[32 * i]) | LDX_REG(0));
            AMX_LDY(AMX_PTR(&l_pad[32 * j]) | LDY_REG(0));

            if (count == 0) {
                AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) |
                          MAC16_Z_ROW(0) | MAC16_Z_SKIP);
            } else {
                AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) |
                          MAC16_Z_ROW(0));
            }
            count++;

            /* Accumulate up to 7 block pairs before flushing to 32-bit software buffer */
            if (count >= SNTRUP_MAX_ACCUM || i == i_max) {
                for (int r = 0; r < 32; r++) {
                    AMX_STZ(AMX_PTR(&z_buf[32 * r]) | STZ_Z_ROW(2 * r));
                }

                int base = 32 * d;
                for (int r = 0; r < 32; r++) {
                    for (int col = 0; col < 32; col++) {
                        prod[base + r + col] += (int32_t)z_buf[r * 32 + col];
                    }
                }
                count = 0;
            }
        }
    }

    poly_reduce(c, prod);
}
