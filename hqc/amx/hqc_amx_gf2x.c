#include "hqc_amx_gf2x.h"
#include "gf2x.h"
#include "aarch64.h"
#include "amx.h"
#include <stdlib.h>
#include <string.h>

/* 
 * Helper to unpack bits to u16.
 * Each GF(2) coefficient is stored as a uint16_t with value 0 or 1.
 */
static void unpack_bits_to_u16(uint16_t *dst, const uint64_t *src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = (src[i / 64] >> (i % 64)) & 1;
    }
}

/* 
 * Helper to pack u16 back to bits (taking mod 2).
 */
static void pack_u16_to_bits(uint64_t *dst, const uint16_t *src, size_t n) {
    memset(dst, 0, GF2X_BYTES(n));
    for (size_t i = 0; i < n; i++) {
        if (src[i] & 1) {
            dst[i / 64] |= (1ULL << (i % 64));
        }
    }
}

/*
 * Dense × Dense polynomial multiplication (AMX integer schoolbook)
 */
void gf2x_mul_amx(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n) {
    /* AMX_SET() must be called by the caller before invoking this function */
    
    size_t padded_n = ((n + 31) / 32) * 32;
    uint16_t *a16 = calloc(padded_n, sizeof(uint16_t));
    uint16_t *b16 = calloc(padded_n, sizeof(uint16_t));
    
    // c16 must be large enough to hold 2*n elements, padded to 32
    size_t c16_size = ((2 * n + 31) / 32) * 32 + 32; 
    uint16_t *c16 = calloc(c16_size, sizeof(uint16_t));
    
    unpack_bits_to_u16(a16, a, n);
    unpack_bits_to_u16(b16, b, n);
    
    size_t nblocks = (n + 31) / 32;
    uint16_t z_buf[32 * 32];
    
    for (size_t i = 0; i < nblocks; i++) {
        // Load block i of a into X register 0
        AMX_LDX(AMX_PTR(&a16[32 * i]) | LDX_REG(0));
        
        for (size_t j = 0; j < nblocks; j++) {
            // Load block j of b into Y register 0
            AMX_LDY(AMX_PTR(&b16[32 * j]) | LDY_REG(0));
            
            // MAC16 outer product. MAC16_Z_SKIP zeroes Z before accumulate.
            // Z[row][col] += X[row] * Y[col]
            AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | 
                      MAC16_Z_ROW(0) | MAC16_Z_SKIP);
            
            // Store the 32x32 Z matrix to memory
            // mac16 with 16-bit values stores row r in Z_ROW(2*r)
            for (int r = 0; r < 32; r++) {
                AMX_STZ(AMX_PTR(&z_buf[32 * r]) | STZ_Z_ROW(2 * r));
            }
            
            // Extract anti-diagonals in software and accumulate into c16
            for (int r = 0; r < 32; r++) {
                for (int col = 0; col < 32; col++) {
                    size_t idx = 32 * i + r + 32 * j + col;
                    c16[idx] += z_buf[r * 32 + col];
                }
            }
        }
    }
    
    // Pack back to bits (taking parity)
    uint64_t *c_unreduced = calloc(GF2X_WORDS(2 * n), sizeof(uint64_t));
    pack_u16_to_bits(c_unreduced, c16, 2 * n - 1);
    
    // Reduce mod (x^n - 1) using existing utility
    gf2x_reduce_xn_minus_1(c, c_unreduced, n);
    
    free(a16);
    free(b16);
    free(c16);
    free(c_unreduced);
}

/*
 * Sparse × Dense polynomial multiplication (AMX vectorized accumulate)
 */
void gf2x_mul_sparse_amx(uint64_t *c, const uint32_t *positions, size_t weight,
                           const uint64_t *b, size_t n) {
    /* AMX_SET() must be called by the caller before invoking this function */
    
    uint16_t *b16 = calloc(n, sizeof(uint16_t));
    size_t c16_size = ((n + 31) / 32) * 32;
    uint16_t *c16 = calloc(c16_size, sizeof(uint16_t));
    
    unpack_bits_to_u16(b16, b, n);
    
    size_t nblocks = (n + 31) / 32;
    uint16_t zero[32] = {0};
    
    for (size_t blk = 0; blk < nblocks; blk++) {
        // Zero Z row 0 by performing a MAC16 outer product with MAC16_Z_SKIP
        AMX_LDX(AMX_PTR(zero) | LDX_REG(7));
        AMX_LDY(AMX_PTR(zero) | LDY_REG(7));
        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(7) | MAC16_Y_REG(7) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);
        
        for (size_t k = 0; k < weight; k++) {
            uint32_t p = positions[k];
            
            // Prepare shifted block in software
            uint16_t temp[32] = {0};
            for (int i = 0; i < 32; i++) {
                size_t idx = blk * 32 + i;
                if (idx < n) {
                    size_t orig = (idx + n - (p % n)) % n;
                    temp[i] = b16[orig];
                }
            }
            
            // Load shifted block into X register 0
            AMX_LDX(AMX_PTR(temp) | LDX_REG(0));
            // Load zeros into Y register 0
            AMX_LDY(AMX_PTR(zero) | LDY_REG(0));
            
            // VECINT ADD: Z_ROW(0) = Z_ROW(0) + X_REG(0) + Y_REG(0)
            AMX_VECINT(VECINT_X_REG(0) | VECINT_Y_REG(0) | VECINT_Z_ROW(0) | VECINT_ALU_MODE_Z_ADD_X_ADD_Y);
        }
        
        // Store the accumulated Z row to c16
        AMX_STZ(AMX_PTR(&c16[blk * 32]) | STZ_Z_ROW(0));
    }
    
    pack_u16_to_bits(c, c16, n);
    
    free(b16);
    free(c16);
}
