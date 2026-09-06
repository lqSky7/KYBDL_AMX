#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include "gf2x.h"
#include "params.h"

// Carry-less multiply two 64-bit values, get 128-bit result
static inline void clmul_64(uint64_t *lo, uint64_t *hi, uint64_t a, uint64_t b) {
    poly128_t result = vmull_p64((poly64_t)a, (poly64_t)b);
    uint64x2_t r = vreinterpretq_u64_p128(result);
    *lo = vgetq_lane_u64(r, 0);
    *hi = vgetq_lane_u64(r, 1);
}

/*
 * Dense × Dense polynomial multiplication using NEON PMULL
 * Uses schoolbook multiplication at the word level
 */
void gf2x_mul_neon(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t words = GF2X_WORDS(n);
    size_t prod_words = GF2X_WORDS(2 * n);
    
    // Allocate product buffer on stack using VLA
    uint64_t prod[prod_words] __attribute__((aligned(16)));
    memset(prod, 0, prod_words * sizeof(uint64_t));
    
    for (size_t i = 0; i < words; i++) {
        for (size_t j = 0; j < words; j++) {
            uint64_t lo, hi;
            clmul_64(&lo, &hi, a[i], b[j]);
            prod[i + j] ^= lo;
            if (i + j + 1 < prod_words) {
                prod[i + j + 1] ^= hi;
            }
        }
    }
    
    gf2x_reduce_xn_minus_1(c, prod, n);
}

/*
 * Helper to XOR-accumulate a shifted polynomial using 128-bit NEON
 */
static void gf2x_xor_shifted_neon(uint64_t *acc, const uint64_t *b,
                                    size_t shift, size_t b_words, size_t acc_words) {
    size_t word_shift = shift / 64;
    unsigned bit_shift = shift % 64;
    
    if (bit_shift == 0) {
        size_t i = 0;
        // Process pairs of words using NEON 128-bit XOR
        for (; i + 1 < b_words; i += 2) {
            if (word_shift + i + 1 < acc_words) {
                uint64x2_t bv = vld1q_u64(&b[i]);
                uint64x2_t av = vld1q_u64(&acc[word_shift + i]);
                vst1q_u64(&acc[word_shift + i], veorq_u64(av, bv));
            } else {
                acc[word_shift + i] ^= b[i];
                if (word_shift + i + 1 < acc_words) {
                    acc[word_shift + i + 1] ^= b[i + 1];
                }
            }
        }
        if (i < b_words && word_shift + i < acc_words) {
            acc[word_shift + i] ^= b[i];
        }
        return;
    }
    
    uint64_t prev = 0;
    size_t i = 0;
    // Process pairs of words using NEON 128-bit XOR
    for (; i + 1 < b_words; i += 2) {
        uint64_t cur0 = b[i];
        uint64_t cur1 = b[i + 1];
        
        uint64_t val0 = (cur0 << bit_shift) | (prev >> (64 - bit_shift));
        uint64_t val1 = (cur1 << bit_shift) | (cur0 >> (64 - bit_shift));
        
        if (word_shift + i + 1 < acc_words) {
            uint64_t vals[2] = {val0, val1};
            uint64x2_t vv = vld1q_u64(vals);
            uint64x2_t av = vld1q_u64(&acc[word_shift + i]);
            vst1q_u64(&acc[word_shift + i], veorq_u64(av, vv));
        } else {
            if (word_shift + i < acc_words) acc[word_shift + i] ^= val0;
            if (word_shift + i + 1 < acc_words) acc[word_shift + i + 1] ^= val1;
        }
        
        prev = cur1;
    }
    
    // Remaining words
    for (; i < b_words; i++) {
        uint64_t cur = b[i];
        uint64_t val = (cur << bit_shift) | (prev >> (64 - bit_shift));
        if (word_shift + i < acc_words) {
            acc[word_shift + i] ^= val;
        }
        prev = cur;
    }
    
    // Leftover bits from the very last word
    if (word_shift + b_words < acc_words) {
        acc[word_shift + b_words] ^= (prev >> (64 - bit_shift));
    }
}

/*
 * Sparse × Dense polynomial multiplication using NEON 128-bit XOR
 */
void gf2x_mul_sparse_neon(uint64_t *c, const uint32_t *positions, size_t weight,
                           const uint64_t *b, size_t n) {
    size_t words = GF2X_WORDS(n);
    size_t prod_words = GF2X_WORDS(2 * n);
    
    // Allocate product buffer on stack using VLA
    uint64_t prod[prod_words] __attribute__((aligned(16)));
    memset(prod, 0, prod_words * sizeof(uint64_t));
    
    for (size_t k = 0; k < weight; k++) {
        // XOR b shifted by positions[k] into prod, using 128-bit NEON XOR
        gf2x_xor_shifted_neon(prod, b, positions[k], words, prod_words);
    }
    
    gf2x_reduce_xn_minus_1(c, prod, n);
}
