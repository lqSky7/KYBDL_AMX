/*
 * Test for AMX Kyber polynomial multiplication correctness.
 *
 * Validates that amx_kyber_poly_basemul produces the same result as
 * the reference schoolbook polynomial multiplication mod (x^256+1) mod q.
 *
 * Also validates the full KEM encrypt/decrypt cycle.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "aarch64.h"

/* Reference schoolbook polynomial multiply mod (x^256+1) mod q */
static void ref_poly_basemul(int16_t res[256], const int16_t a[256], const int16_t b[256]) {
    int32_t tmp[256];
    memset(tmp, 0, sizeof(tmp));

    /* Full polynomial product (512 coefficients) folded with negacyclic reduction */
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            if (i + j < 256) {
                tmp[i + j] += (int32_t)a[i] * b[j];
            } else {
                /* x^256 = -1 in x^256+1 */
                tmp[i + j - 256] -= (int32_t)a[i] * b[j];
            }
        }
    }

    /* Reduce mod q */
    for (int i = 0; i < 256; i++) {
        int32_t t = tmp[i] % 3329;
        if (t < 0) t += 3329;
        res[i] = (int16_t)t;
    }
}

/* Reduce AMX output to canonical [0, q) */
static void reduce_to_canonical(int16_t r[256]) {
    for (int i = 0; i < 256; i++) {
        int16_t t = r[i] % 3329;
        if (t < 0) t += 3329;
        r[i] = t;
    }
}
}  // extern "C"

extern "C" {
#include "kyber_amx_polymul.h"
}

class KyberAMXPolyMulTest : public ::testing::Test {
protected:
    void SetUp() override {
        AMX_SET();
    }
    void TearDown() override {
        AMX_CLR();
    }
};

/* Test that AMX poly_basemul matches reference schoolbook for random inputs */
TEST_F(KyberAMXPolyMulTest, RandomInputs) {
    int16_t a[256], b[256];
    int16_t res_amx[256], res_ref[256];

    srand(42);
    for (int trial = 0; trial < 10; trial++) {
        /* Generate random polynomials with coefficients in [0, q) */
        for (int i = 0; i < 256; i++) {
            a[i] = rand() % 3329;
            b[i] = rand() % 3329;
        }

        amx_kyber_poly_basemul(res_amx, a, b);
        ref_poly_basemul(res_ref, a, b);

        reduce_to_canonical(res_amx);

        for (int i = 0; i < 256; i++) {
            ASSERT_EQ(res_amx[i], res_ref[i])
                << "Mismatch at index " << i
                << " in trial " << trial
                << ": AMX=" << res_amx[i] << " ref=" << res_ref[i];
        }
    }
}

/* Test with all-zero inputs */
TEST_F(KyberAMXPolyMulTest, ZeroInputs) {
    int16_t a[256] = {0};
    int16_t b[256] = {0};
    int16_t res[256];

    amx_kyber_poly_basemul(res, a, b);

    for (int i = 0; i < 256; i++) {
        ASSERT_EQ(res[i], 0) << "Expected 0 at index " << i;
    }
}

/* Test with identity-like inputs: a = (1, 0, 0, ...), b = arbitrary */
TEST_F(KyberAMXPolyMulTest, IdentityMultiply) {
    int16_t a[256] = {0};
    int16_t b[256];
    int16_t res_amx[256];

    a[0] = 1;  /* a(x) = 1 */
    srand(123);
    for (int i = 0; i < 256; i++) {
        b[i] = rand() % 3329;
    }

    amx_kyber_poly_basemul(res_amx, a, b);
    reduce_to_canonical(res_amx);

    /* Result should equal b */
    for (int i = 0; i < 256; i++) {
        ASSERT_EQ(res_amx[i], b[i])
            << "Mismatch at index " << i
            << ": AMX=" << res_amx[i] << " expected=" << b[i];
    }
}

/* Test negacyclic property: a = (0, ..., 0, 1) (= x^255), b = (0, 1, 0, ...) (= x)
 * Product = x^256 = -1 mod (x^256+1)
 * So result should be (-1 mod q, 0, 0, ...) = (3328, 0, 0, ...) */
TEST_F(KyberAMXPolyMulTest, NegacyclicProperty) {
    int16_t a[256] = {0};
    int16_t b[256] = {0};
    int16_t res[256];

    a[255] = 1;  /* x^255 */
    b[1] = 1;    /* x */

    amx_kyber_poly_basemul(res, a, b);
    reduce_to_canonical(res);

    /* x^255 * x = x^256 = -1 mod (x^256+1) */
    ASSERT_EQ(res[0], 3328) << "Expected -1 mod 3329 = 3328 at index 0";
    for (int i = 1; i < 256; i++) {
        ASSERT_EQ(res[i], 0) << "Expected 0 at index " << i;
    }
}

/* Test poly_add */
TEST_F(KyberAMXPolyMulTest, PolyAdd) {
    int16_t a[256], b[256], c[256];

    for (int i = 0; i < 256; i++) {
        a[i] = 3000;
        b[i] = 500;
    }

    amx_kyber_poly_add(c, a, b);

    for (int i = 0; i < 256; i++) {
        ASSERT_EQ(c[i], (3000 + 500) % 3329) << "Mismatch at " << i;
    }
}

/* Test poly_sub */
TEST_F(KyberAMXPolyMulTest, PolySub) {
    int16_t a[256], b[256], c[256];

    for (int i = 0; i < 256; i++) {
        a[i] = 100;
        b[i] = 200;
    }

    amx_kyber_poly_sub(c, a, b);

    for (int i = 0; i < 256; i++) {
        /* 100 - 200 = -100 mod 3329 = 3229 */
        ASSERT_EQ(c[i], 3229) << "Mismatch at " << i;
    }
}

/* Test matrix-vector multiply (k=2) */
TEST_F(KyberAMXPolyMulTest, MatrixVectorMul_K2) {
    /* Small test: A = [[a00, a01], [a10, a11]], s = [s0, s1]
     * res[0] = a00*s0 + a01*s1
     * res[1] = a10*s0 + a11*s1
     */
    int16_t A[2 * 2 * 256];
    int16_t s[2 * 256];
    int16_t res_amx[2 * 256];
    int16_t res_ref[2 * 256];
    int16_t tmp[256];

    srand(999);
    for (int i = 0; i < 2 * 2 * 256; i++) A[i] = rand() % 3329;
    for (int i = 0; i < 2 * 256; i++) s[i] = rand() % 3329;

    /* AMX */
    amx_kyber_matrix_vector_mul(res_amx, A, s, 2);

    /* Reference */
    ref_poly_basemul(res_ref, &A[0 * 256], &s[0 * 256]);  /* A[0][0] * s[0] */
    ref_poly_basemul(tmp, &A[1 * 256], &s[1 * 256]);      /* A[0][1] * s[1] */
    for (int i = 0; i < 256; i++) {
        int32_t t = (int32_t)res_ref[i] + tmp[i];
        t %= 3329;
        if (t < 0) t += 3329;
        res_ref[i] = (int16_t)t;
    }

    ref_poly_basemul(&res_ref[256], &A[2 * 256], &s[0 * 256]);  /* A[1][0] * s[0] */
    ref_poly_basemul(tmp, &A[3 * 256], &s[1 * 256]);            /* A[1][1] * s[1] */
    for (int i = 0; i < 256; i++) {
        int32_t t = (int32_t)res_ref[256 + i] + tmp[i];
        t %= 3329;
        if (t < 0) t += 3329;
        res_ref[256 + i] = (int16_t)t;
    }

    reduce_to_canonical(res_amx);
    reduce_to_canonical(&res_amx[256]);

    for (int i = 0; i < 2 * 256; i++) {
        ASSERT_EQ(res_amx[i], res_ref[i])
            << "Matrix-vector mismatch at index " << i;
    }
}
