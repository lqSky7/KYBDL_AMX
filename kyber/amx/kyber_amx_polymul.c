/*
 * AMX / NEON Polynomial Multiplication for Kyber (q=3329, n=256, mod x^256+1)
 *
 * Computes c(x) = a(x) * b(x) mod (x^256 + 1) mod q, where q=3329.
 *
 * amx_kyber_poly_basemul is genuinely AMX-accelerated: it decomposes each
 * operand into base-8 limb polynomials and drives the existing MAC16-based
 * negacyclic TMVP kernel from amx/polymodmul_matmul.c once per limb pair,
 * recombining the exact results in 64-bit arithmetic before a final
 * reduction mod q. See the detailed rationale above amx_kyber_poly_basemul
 * for why this is needed instead of directly reusing Saber's free mod-2^16
 * trick (Kyber's prime modulus doesn't divide 2^16 the way Saber's does).
 *
 * License: CC0 1.0 Universal / MIT
 */

#include "kyber_amx_polymul.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "aarch64.h"
#include "amx.h"
#include "polymodmul.h"

#define BARRETT_V 20159  /* round(2^26 / 3329) */
#define KYBER_Q_VAL 3329

/* Barrett reduction for a single integer to canonical range [0, q) */
static inline int16_t barrett_reduce_single(int32_t a) {
    int32_t t = ((int64_t)BARRETT_V * a + (1 << 25)) >> 26;
    int16_t res = (int16_t)(a - t * KYBER_Q_VAL);
    if (res < 0) res += KYBER_Q_VAL;
    if (res >= KYBER_Q_VAL) res -= KYBER_Q_VAL;
    return res;
}

void poly_reduce(int16_t c[KYBER_AMX_N]) {
    for (int i = 0; i < KYBER_AMX_N; i++) {
        c[i] = barrett_reduce_single(c[i]);
    }
}

void kyber_barrett_reduce_vec(int16_t *res, const int32_t *a, int n) {
    for (int i = 0; i < n; i++) {
        res[i] = barrett_reduce_single(a[i]);
    }
}

/* ========================================================================
 * AMX TMVP Polynomial Multiplication: c = a * b mod (x^256+1) mod q
 *
 * Real hardware background: AMX's MAC16 instruction accumulates strictly
 * modulo 2^16 (that's what "mod_65536" in amx_poly_mul_mod_65536_mod_x_d_
 * plus_1_u16_32nx32n_coeffs literally means, and it's exact for ANY
 * accumulated magnitude - modular addition is exact regardless of overflow).
 * That's "free" for Saber, whose q is a power of two dividing 2^16, since
 * reducing an exact-mod-2^16 value further mod a power-of-two q is just a
 * bitmask. It is NOT free for Kyber's prime q=3329, because
 * gcd(65536, 3329) = 1: knowing a sum mod 65536 tells us nothing about that
 * sum mod q without also knowing the exact (unreduced) integer value.
 *
 * To get an AMX-accelerated *exact* integer convolution (which we can then
 * Barrett-reduce mod q), we decompose each coefficient into four base-8
 * (3-bit) limbs: a = a3*8^3 + a2*8^2 + a1*8 + a0, with each limb in [0,8).
 * Kyber coefficients are < 4096 = 8^4, so four limbs always suffice.
 *
 * For any pair of limb polynomials (values in [0,8)), each output
 * coefficient of their exact negacyclic convolution is bounded by
 * 256 * 7 * 7 = 12544 < 32768. That means the mod-65536 result AMX
 * produces for that limb-pair convolution, reinterpreted as a signed
 * int16, is *exactly* the true (unreduced) value - no ambiguity, because
 * the true value can never leave the int16 range representable mod 2^16.
 *
 * We therefore run the existing, unmodified Saber/FrodoKEM-style AMX TMVP
 * kernel (amx_poly_mul_mod_65536_mod_x_d_plus_1_u16_32nx32n_coeffs, from
 * amx/polymodmul_matmul.c) once per limb pair (16 calls total), and
 * recombine the 16 exact small sub-convolutions with integer weights
 * 8^(p+q) in 64-bit arithmetic to reconstruct the exact integer
 * convolution, before a final Barrett-style reduction mod q.
 *
 * This is ~16x the AMX work of a single Saber polymul call, but is still
 * expected to be far cheaper than the O(256^2) scalar schoolbook multiply
 * (65536 scalar mul-adds plus hundreds of modulo operations) it replaces.
 * Real cycle counts should be re-measured once this lands.
 *
 * Inputs a, b are expected to have coefficients representable mod q; they
 * are normalized to [0, q) internally. Output coefficients are reduced to
 * [0, q).
 * ========================================================================*/
#define KYBER_AMX_LIMB_BITS 3
#define KYBER_AMX_LIMB_BASE (1 << KYBER_AMX_LIMB_BITS) /* 8 */
#define KYBER_AMX_LIMB_MASK (KYBER_AMX_LIMB_BASE - 1)  /* 7 */
#define KYBER_AMX_NUM_LIMBS 4 /* 4 limbs of 3 bits cover [0, 4096) > q */

/*
 * Splits a polynomial with coefficients in [0, q) into KYBER_AMX_NUM_LIMBS
 * base-8 digit polynomials, each with coefficients in [0, 8). limbs[p][k] is
 * the p-th least-significant digit of poly[k].
 */
static void kyber_amx_split_limbs(uint16_t limbs[KYBER_AMX_NUM_LIMBS][KYBER_AMX_N],
                                  const int16_t poly[KYBER_AMX_N]) {
    for (int k = 0; k < KYBER_AMX_N; k++) {
        uint16_t v = (uint16_t)poly[k];
        for (int p = 0; p < KYBER_AMX_NUM_LIMBS; p++) {
            limbs[p][k] = v & KYBER_AMX_LIMB_MASK;
            v >>= KYBER_AMX_LIMB_BITS;
        }
    }
}

void amx_kyber_poly_basemul(int16_t res[KYBER_AMX_N],
                            const int16_t a[KYBER_AMX_N],
                            const int16_t b[KYBER_AMX_N]) {
    /* Normalize inputs to non-negative [0, q) */
    int16_t a_norm[KYBER_AMX_N], b_norm[KYBER_AMX_N];
    for (int i = 0; i < KYBER_AMX_N; i++) {
        int16_t ta = a[i] % KYBER_Q_VAL;
        if (ta < 0) ta += KYBER_Q_VAL;
        a_norm[i] = ta;

        int16_t tb = b[i] % KYBER_Q_VAL;
        if (tb < 0) tb += KYBER_Q_VAL;
        b_norm[i] = tb;
    }

    /* Decompose both operands into 4 base-8 limb polynomials each. */
    uint16_t a_limbs[KYBER_AMX_NUM_LIMBS][KYBER_AMX_N];
    uint16_t b_limbs[KYBER_AMX_NUM_LIMBS][KYBER_AMX_N];
    kyber_amx_split_limbs(a_limbs, a_norm);
    kyber_amx_split_limbs(b_limbs, b_norm);

    /*
     * Exact (unreduced) integer convolution accumulator. Bounded by
     * n * (q-1)^2 ~ 2.84e9 in magnitude, so int64_t is used to be safe.
     */
    int64_t exact[KYBER_AMX_N];
    memset(exact, 0, sizeof(exact));

    uint16_t sub_xy[KYBER_AMX_N];
    int64_t weight = 1;

    for (int w = 0; w < 2 * KYBER_AMX_NUM_LIMBS - 1; w++) {
        /* All (p, q) limb-index pairs contributing weight 8^w. */
        int p_lo = (w >= KYBER_AMX_NUM_LIMBS) ? (w - KYBER_AMX_NUM_LIMBS + 1) : 0;
        int p_hi = (w < KYBER_AMX_NUM_LIMBS) ? w : (KYBER_AMX_NUM_LIMBS - 1);

        for (int p = p_lo; p <= p_hi; p++) {
            int qidx = w - p;

            /*
             * Existing, unmodified AMX TMVP kernel (Saber/FrodoKEM pattern):
             * computes the exact negacyclic convolution mod 2^16. Because
             * limb values are in [0, 8), each output coefficient's true
             * magnitude is <= 256 * 7 * 7 = 12544 < 32768, so reinterpreting
             * the mod-65536 result as signed int16 recovers it exactly.
             */
            amx_poly_mul_mod_65536_mod_x_d_plus_1_u16_32nx32n_coeffs(
                sub_xy, a_limbs[p], b_limbs[qidx], KYBER_AMX_N - 1, KYBER_AMX_N / 32);

            for (int k = 0; k < KYBER_AMX_N; k++) {
                exact[k] += weight * (int16_t)sub_xy[k];
            }
        }

        weight *= KYBER_AMX_LIMB_BASE;
    }

    /* Final reduction mod q (exact values, so plain modulo suffices). */
    for (int i = 0; i < KYBER_AMX_N; i++) {
        int64_t t = exact[i] % KYBER_Q_VAL;
        if (t < 0) t += KYBER_Q_VAL;
        res[i] = (int16_t)t;
    }
}

/* ========================================================================
 * AMX Matrix-Vector Multiplication for Kyber
 * ========================================================================*/
void amx_kyber_matrix_vector_mul(int16_t *res,
                                 const int16_t *a,
                                 const int16_t *s,
                                 int k) {
    int16_t tmp[KYBER_AMX_N];

    for (int i = 0; i < k; i++) {
        /* First multiply: res[i] = A[i][0] * s[0] */
        amx_kyber_poly_basemul(&res[i * KYBER_AMX_N],
                               &a[(i * k + 0) * KYBER_AMX_N],
                               &s[0 * KYBER_AMX_N]);

        /* Accumulate remaining: res[i] += A[i][j] * s[j] */
        for (int j = 1; j < k; j++) {
            amx_kyber_poly_basemul(tmp,
                                   &a[(i * k + j) * KYBER_AMX_N],
                                   &s[j * KYBER_AMX_N]);

            amx_kyber_poly_add(&res[i * KYBER_AMX_N],
                               &res[i * KYBER_AMX_N],
                               tmp);
        }
    }
}

/* ========================================================================
 * AMX Inner Product for Kyber decapsulation
 * ========================================================================*/
void amx_kyber_inner_prod(int16_t res[KYBER_AMX_N],
                          const int16_t *b,
                          const int16_t *s,
                          int k) {
    int16_t tmp[KYBER_AMX_N];

    amx_kyber_poly_basemul(res, &b[0], &s[0]);

    for (int j = 1; j < k; j++) {
        amx_kyber_poly_basemul(tmp,
                               &b[j * KYBER_AMX_N],
                               &s[j * KYBER_AMX_N]);
        amx_kyber_poly_add(res, res, tmp);
    }
}

/* ========================================================================
 * Polynomial addition and subtraction mod q
 * ========================================================================*/
void amx_kyber_poly_add(int16_t c[KYBER_AMX_N],
                        const int16_t a[KYBER_AMX_N],
                        const int16_t b[KYBER_AMX_N]) {
    for (int i = 0; i < KYBER_AMX_N; i++) {
        int32_t sum = (int32_t)a[i] + b[i];
        if (sum >= KYBER_Q_VAL) sum -= KYBER_Q_VAL;
        c[i] = (int16_t)sum;
    }
}

void amx_kyber_poly_sub(int16_t c[KYBER_AMX_N],
                        const int16_t a[KYBER_AMX_N],
                        const int16_t b[KYBER_AMX_N]) {
    for (int i = 0; i < KYBER_AMX_N; i++) {
        int32_t diff = (int32_t)a[i] - b[i];
        if (diff < 0) diff += KYBER_Q_VAL;
        c[i] = (int16_t)diff;
    }
}
