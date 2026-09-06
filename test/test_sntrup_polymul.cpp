/*
 * sntrup761 Polynomial Multiplication — Correctness & Benchmark Tests
 *
 * Tests polynomial multiplication in Z_4591[x] / (x^761 - x - 1)
 * across three implementations: Reference C, NEON, and AMX.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "sntrup761/poly.h"
#include "sntrup761/amx/sntrup_amx_polymul.h"
#include "amx/aarch64.h"
}

class SntrupPolyTest : public ::testing::Test {
protected:
    void SetUp() override {
        AMX_SET();
    }
    void TearDown() override {
        AMX_CLR();
    }
};

/* Fill polynomial with deterministic pseudo-random coefficients mod q */
static void random_poly(int16_t *p, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        int r = rand() % SNTRUP_Q;
        if (r > SNTRUP_Q_HALF) r -= SNTRUP_Q;
        p[i] = (int16_t)r;
    }
}

/* Fill polynomial with ternary coefficients {-1, 0, 1} */
static void random_ternary(int16_t *p, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        p[i] = (int16_t)((rand() % 3) - 1);
    }
}

static int poly_equal(const int16_t *a, const int16_t *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

/* ========================================================================
 * Correctness Tests — Small Polynomial (hand-verifiable)
 * ======================================================================== */

/*
 * Multiply by zero: a * 0 = 0
 */
TEST_F(SntrupPolyTest, MulByZero) {
    std::vector<int16_t> a(SNTRUP_P), zero_poly(SNTRUP_P, 0), c(SNTRUP_P);
    random_poly(a.data(), SNTRUP_P, 42);

    poly_mul_ref(c.data(), a.data(), zero_poly.data());
    for (int i = 0; i < SNTRUP_P; i++) {
        ASSERT_EQ(c[i], 0) << "Ref: a * 0 != 0 at coefficient " << i;
    }
}

/*
 * Multiply by 1 (the constant polynomial 1): a * 1 = a
 */
TEST_F(SntrupPolyTest, MulByOne) {
    std::vector<int16_t> a(SNTRUP_P), one(SNTRUP_P, 0), c(SNTRUP_P);
    random_poly(a.data(), SNTRUP_P, 42);
    one[0] = 1;

    poly_mul_ref(c.data(), a.data(), one.data());

    EXPECT_TRUE(poly_equal(a.data(), c.data(), SNTRUP_P))
        << "Ref: a * 1 != a";
}

/*
 * Small hand-verifiable test: multiply (1 + x) * (1 + x) mod (x^p - x - 1)
 * Expected: (1 + x)^2 = 1 + 2x + x^2
 * In Z_4591: coefficients are [1, 2, 1, 0, 0, ...]
 */
TEST_F(SntrupPolyTest, SmallSquare) {
    std::vector<int16_t> a(SNTRUP_P, 0), c(SNTRUP_P);
    a[0] = 1; a[1] = 1;

    poly_mul_ref(c.data(), a.data(), a.data());

    EXPECT_EQ(c[0], 1) << "(1+x)^2 coeff 0";
    EXPECT_EQ(c[1], 2) << "(1+x)^2 coeff 1";
    EXPECT_EQ(c[2], 1) << "(1+x)^2 coeff 2";
    for (int i = 3; i < SNTRUP_P; i++) {
        EXPECT_EQ(c[i], 0) << "(1+x)^2 coeff " << i;
    }
}

/*
 * Test ring reduction: multiply by x^{p-1}.
 * a = 1 (constant), b = x^{p-1}
 * Product = x^{p-1} (no reduction needed, degree < p)
 */
TEST_F(SntrupPolyTest, MulByXpMinus1) {
    std::vector<int16_t> a(SNTRUP_P, 0), b(SNTRUP_P, 0), c(SNTRUP_P);
    a[0] = 1;
    b[SNTRUP_P - 1] = 1;  /* x^760 */

    poly_mul_ref(c.data(), a.data(), b.data());

    for (int i = 0; i < SNTRUP_P; i++) {
        int16_t expected = (i == SNTRUP_P - 1) ? 1 : 0;
        EXPECT_EQ(c[i], expected) << "1 * x^(p-1) coeff " << i;
    }
}

/*
 * Test ring reduction: multiply x * x^{p-1} = x^p ≡ x + 1 mod (x^p - x - 1)
 */
TEST_F(SntrupPolyTest, RingReduction) {
    std::vector<int16_t> a(SNTRUP_P, 0), b(SNTRUP_P, 0), c(SNTRUP_P);
    a[1] = 1;             /* x */
    b[SNTRUP_P - 1] = 1;  /* x^{p-1} */

    poly_mul_ref(c.data(), a.data(), b.data());

    /* x^p = x + 1 */
    EXPECT_EQ(c[0], 1) << "x^p coeff 0 should be 1";
    EXPECT_EQ(c[1], 1) << "x^p coeff 1 should be 1";
    for (int i = 2; i < SNTRUP_P; i++) {
        EXPECT_EQ(c[i], 0) << "x^p coeff " << i << " should be 0";
    }
}

/* ========================================================================
 * Cross-Implementation Agreement Tests
 * ======================================================================== */

TEST_F(SntrupPolyTest, NeonMatchesRef) {
    std::vector<int16_t> a(SNTRUP_P), b(SNTRUP_P), c_ref(SNTRUP_P), c_neon(SNTRUP_P);
    random_poly(a.data(), SNTRUP_P, 1111);
    random_poly(b.data(), SNTRUP_P, 2222);

    poly_mul_ref(c_ref.data(), a.data(), b.data());
    poly_mul_neon(c_neon.data(), a.data(), b.data());

    EXPECT_TRUE(poly_equal(c_ref.data(), c_neon.data(), SNTRUP_P))
        << "NEON result differs from reference";
}

TEST_F(SntrupPolyTest, AmxMatchesRef) {
    std::vector<int16_t> a(SNTRUP_P), b(SNTRUP_P), c_ref(SNTRUP_P), c_amx(SNTRUP_P);
    random_poly(a.data(), SNTRUP_P, 3333);
    random_poly(b.data(), SNTRUP_P, 4444);

    poly_mul_ref(c_ref.data(), a.data(), b.data());
    poly_mul_amx(c_amx.data(), a.data(), b.data());

    EXPECT_TRUE(poly_equal(c_ref.data(), c_amx.data(), SNTRUP_P))
        << "AMX result differs from reference";
}

/*
 * Commutativity: a * b = b * a
 */
TEST_F(SntrupPolyTest, Commutativity) {
    std::vector<int16_t> a(SNTRUP_P), b(SNTRUP_P), ab(SNTRUP_P), ba(SNTRUP_P);
    random_poly(a.data(), SNTRUP_P, 5555);
    random_poly(b.data(), SNTRUP_P, 6666);

    poly_mul_ref(ab.data(), a.data(), b.data());
    poly_mul_ref(ba.data(), b.data(), a.data());

    EXPECT_TRUE(poly_equal(ab.data(), ba.data(), SNTRUP_P))
        << "Commutativity failed: a*b != b*a";
}

/*
 * Ternary × general: all three implementations agree
 */
TEST_F(SntrupPolyTest, TernaryMulAgreement) {
    std::vector<int16_t> r(SNTRUP_P), h(SNTRUP_P);
    std::vector<int16_t> c_ref(SNTRUP_P), c_neon(SNTRUP_P), c_amx(SNTRUP_P);
    random_ternary(r.data(), SNTRUP_P, 7777);
    random_poly(h.data(), SNTRUP_P, 8888);

    poly_mul_ref(c_ref.data(), r.data(), h.data());
    poly_mul_neon(c_neon.data(), r.data(), h.data());
    poly_mul_amx(c_amx.data(), r.data(), h.data());

    EXPECT_TRUE(poly_equal(c_ref.data(), c_neon.data(), SNTRUP_P))
        << "NEON ternary mul differs from reference";
    EXPECT_TRUE(poly_equal(c_ref.data(), c_amx.data(), SNTRUP_P))
        << "AMX ternary mul differs from reference";
}

/* ========================================================================
 * Benchmarks
 * ======================================================================== */

TEST_F(SntrupPolyTest, BenchmarkDenseMul) {
    std::vector<int16_t> a(SNTRUP_P), b(SNTRUP_P), c(SNTRUP_P);
    random_poly(a.data(), SNTRUP_P, 1234);
    random_poly(b.data(), SNTRUP_P, 5678);

    const int WARMUP = 2;
    const int ITERS = 20;

    /* Reference C */
    for (int i = 0; i < WARMUP; i++) poly_mul_ref(c.data(), a.data(), b.data());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) poly_mul_ref(c.data(), a.data(), b.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

    /* NEON */
    for (int i = 0; i < WARMUP; i++) poly_mul_neon(c.data(), a.data(), b.data());
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) poly_mul_neon(c.data(), a.data(), b.data());
    t1 = std::chrono::high_resolution_clock::now();
    double neon_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

    /* AMX */
    for (int i = 0; i < WARMUP; i++) poly_mul_amx(c.data(), a.data(), b.data());
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) poly_mul_amx(c.data(), a.data(), b.data());
    t1 = std::chrono::high_resolution_clock::now();
    double amx_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

    printf("\n");
    printf("========================================================================\n");
    printf("sntrup761 DENSE x DENSE POLYNOMIAL MULTIPLICATION\n");
    printf("Ring: Z_%d[x] / (x^%d - x - 1)\n", SNTRUP_Q, SNTRUP_P);
    printf("------------------------------------------------------------------------\n");
    printf("  Reference C (schoolbook) : %10.1f us\n", ref_us);
    printf("  NEON vmlal (schoolbook)  : %10.1f us\n", neon_us);
    printf("  AMX mac16 (block outer)  : %10.1f us\n", amx_us);
    printf("------------------------------------------------------------------------\n");
    printf("  Speedup NEON vs Ref      : %8.2fx\n", ref_us / neon_us);
    printf("  Speedup AMX  vs Ref      : %8.2fx\n", ref_us / amx_us);
    printf("  Speedup AMX  vs NEON     : %8.2fx\n", neon_us / amx_us);
    printf("========================================================================\n");
}

TEST_F(SntrupPolyTest, BenchmarkTernaryMul) {
    std::vector<int16_t> r(SNTRUP_P), h(SNTRUP_P), c(SNTRUP_P);
    random_ternary(r.data(), SNTRUP_P, 9012);
    random_poly(h.data(), SNTRUP_P, 3456);

    const int WARMUP = 2;
    const int ITERS = 20;

    /* Reference C */
    for (int i = 0; i < WARMUP; i++) poly_mul_ref(c.data(), r.data(), h.data());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) poly_mul_ref(c.data(), r.data(), h.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

    /* NEON */
    for (int i = 0; i < WARMUP; i++) poly_mul_neon(c.data(), r.data(), h.data());
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) poly_mul_neon(c.data(), r.data(), h.data());
    t1 = std::chrono::high_resolution_clock::now();
    double neon_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

    /* AMX */
    for (int i = 0; i < WARMUP; i++) poly_mul_amx(c.data(), r.data(), h.data());
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) poly_mul_amx(c.data(), r.data(), h.data());
    t1 = std::chrono::high_resolution_clock::now();
    double amx_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;

    printf("\n");
    printf("========================================================================\n");
    printf("sntrup761 TERNARY x DENSE POLYNOMIAL MULTIPLICATION\n");
    printf("Ring: Z_%d[x] / (x^%d - x - 1), ternary r in {-1, 0, 1}\n",
           SNTRUP_Q, SNTRUP_P);
    printf("------------------------------------------------------------------------\n");
    printf("  Reference C (schoolbook) : %10.1f us\n", ref_us);
    printf("  NEON vmlal (schoolbook)  : %10.1f us\n", neon_us);
    printf("  AMX mac16 (block outer)  : %10.1f us\n", amx_us);
    printf("------------------------------------------------------------------------\n");
    printf("  Speedup NEON vs Ref      : %8.2fx\n", ref_us / neon_us);
    printf("  Speedup AMX  vs Ref      : %8.2fx\n", ref_us / amx_us);
    printf("  Speedup AMX  vs NEON     : %8.2fx\n", neon_us / amx_us);
    printf("========================================================================\n");
}
