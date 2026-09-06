#include "gf2x.h"
#include <string.h>

void gf2x_zero(uint64_t *a, size_t words) {
    memset(a, 0, words * sizeof(uint64_t));
}

void gf2x_copy(uint64_t *dst, const uint64_t *src, size_t words) {
    memcpy(dst, src, words * sizeof(uint64_t));
}

void gf2x_add(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t words) {
    for (size_t i = 0; i < words; i++) {
        c[i] = a[i] ^ b[i];
    }
}

int gf2x_equal(const uint64_t *a, const uint64_t *b, size_t words) {
    for (size_t i = 0; i < words; i++) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

void gf2x_xor_shifted(uint64_t *acc, const uint64_t *b, size_t shift, size_t n, size_t acc_words) {
    size_t shift_words = shift / 64;
    size_t shift_bits = shift % 64;
    size_t b_words = GF2X_WORDS(n);
    
    if (shift_bits == 0) {
        for (size_t i = 0; i < b_words; i++) {
            if (shift_words + i < acc_words) {
                acc[shift_words + i] ^= b[i];
            }
        }
    } else {
        for (size_t i = 0; i < b_words; i++) {
            if (shift_words + i < acc_words) {
                acc[shift_words + i] ^= (b[i] << shift_bits);
            }
            if (shift_words + i + 1 < acc_words) {
                acc[shift_words + i + 1] ^= (b[i] >> (64 - shift_bits));
            }
        }
    }
}

void gf2x_reduce_xn_minus_1(uint64_t *c, const uint64_t *prod, size_t n) {
    size_t n_words = GF2X_WORDS(n);
    size_t prod_words = GF2X_WORDS(2 * n);

    /* Start with the lower n bits of prod */
    gf2x_copy(c, prod, n_words);

    /* Mask the last word of c to exactly n bits */
    size_t bit_offset = n % 64;
    if (bit_offset != 0) {
        c[n_words - 1] &= (1ULL << bit_offset) - 1;
    }

    /*
     * XOR the upper portion (bits n .. 2n-2) back into positions 0 .. n-2.
     * The upper portion starts at bit position n in prod, which is
     * word_start = n/64, with bit_start = n%64 bits into that word.
     *
     * We extract words from the upper portion with appropriate shifting
     * and XOR them into c word-by-word.
     */
    size_t word_start = n / 64;
    size_t bit_start = n % 64;

    if (bit_start == 0) {
        /* Aligned case: upper portion starts at a word boundary */
        for (size_t i = 0; i < n_words && (word_start + i) < prod_words; i++) {
            c[i] ^= prod[word_start + i];
        }
    } else {
        /* Unaligned case: upper portion is shifted within words */
        for (size_t i = 0; i < n_words; i++) {
            uint64_t upper_word = 0;
            size_t src = word_start + i;
            if (src < prod_words) {
                upper_word = prod[src] >> bit_start;
            }
            if (src + 1 < prod_words) {
                upper_word |= prod[src + 1] << (64 - bit_start);
            }
            c[i] ^= upper_word;
        }
    }

    /* Re-mask the last word to exactly n bits */
    if (bit_offset != 0) {
        c[n_words - 1] &= (1ULL << bit_offset) - 1;
    }
}

void gf2x_mul_ref(uint64_t *c, const uint64_t *a, const uint64_t *b, size_t n) {
    size_t prod_words = GF2X_WORDS(2 * n);
    uint64_t prod[prod_words];
    gf2x_zero(prod, prod_words);
    
    for (size_t i = 0; i < n; i++) {
        if (gf2x_get_bit(a, i)) {
            gf2x_xor_shifted(prod, b, i, n, prod_words);
        }
    }
    
    gf2x_reduce_xn_minus_1(c, prod, n);
}

void gf2x_mul_sparse_ref(uint64_t *c, const uint32_t *positions, size_t weight, const uint64_t *b, size_t n) {
    size_t prod_words = GF2X_WORDS(2 * n);
    uint64_t prod[prod_words];
    gf2x_zero(prod, prod_words);
    
    for (size_t i = 0; i < weight; i++) {
        uint32_t pos = positions[i];
        if (pos < n) {
            gf2x_xor_shifted(prod, b, pos, n, prod_words);
        }
    }
    
    gf2x_reduce_xn_minus_1(c, prod, n);
}
