/*
 * GF(2)[X] Polynomial Multiplication — Correctness & Benchmark Tests
 *
 * Three-way comparison: Reference C vs NEON PMULL vs AMX
 * Tests both dense×dense and sparse×dense multiplication.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

extern "C" {
#include "params.h"
#include "gf2x.h"
#include "aarch64.h"
}

/* ========================================================================
 * Test Fixture
 * ======================================================================== */

class GF2XTest : public ::testing::Test {
protected:
    void SetUp() override {
        AMX_SET();
    }

    /* Fill polynomial with random bits */
    static void random_poly(uint64_t *a, size_t n, unsigned seed) {
        srand(seed);
        size_t words = GF2X_WORDS(n);
        for (size_t i = 0; i < words; i++) {
            a[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        }
        /* Mask out bits >= n in the last word */
        size_t tail = n % 64;
        if (tail > 0) {
            a[words - 1] &= (1ULL << tail) - 1;
        }
    }

    /* Generate random sparse positions */
    static void random_sparse(uint32_t *positions, size_t weight, size_t n, unsigned seed) {
        srand(seed);
        /* Simple reservoir sampling */
        std::vector<uint32_t> all_pos(n);
        for (uint32_t i = 0; i < (uint32_t)n; i++) all_pos[i] = i;
        for (size_t i = 0; i < weight; i++) {
            size_t j = i + (rand() % (n - i));
            std::swap(all_pos[i], all_pos[j]);
        }
        std::sort(all_pos.begin(), all_pos.begin() + weight);
        for (size_t i = 0; i < weight; i++) {
            positions[i] = all_pos[i];
        }
    }

    /* Convert sparse positions to dense packed polynomial */
    static void sparse_to_dense(uint64_t *a, const uint32_t *positions, size_t weight, size_t n) {
        gf2x_zero(a, GF2X_WORDS(n));
        for (size_t i = 0; i < weight; i++) {
            gf2x_set_bit(a, positions[i]);
        }
    }
};

/* ========================================================================
 * Utility Function Tests
 * ======================================================================== */

TEST_F(GF2XTest, ZeroAndCopy) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    std::vector<uint64_t> a(words), b(words);

    random_poly(a.data(), n, 42);
    gf2x_zero(b.data(), words);

    /* b should be all zeros */
    for (size_t i = 0; i < words; i++) {
        EXPECT_EQ(b[i], 0ULL);
    }

    /* Copy a to b, they should be equal */
    gf2x_copy(b.data(), a.data(), words);
    EXPECT_TRUE(gf2x_equal(a.data(), b.data(), words));
}

TEST_F(GF2XTest, AddSelfIsZero) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    std::vector<uint64_t> a(words), c(words);

    random_poly(a.data(), n, 123);
    gf2x_add(c.data(), a.data(), a.data(), words);

    /* a XOR a = 0 */
    for (size_t i = 0; i < words; i++) {
        EXPECT_EQ(c[i], 0ULL);
    }
}

/* ========================================================================
 * Dense × Dense Correctness Tests
 * ======================================================================== */

/* Small-n test where we can verify by hand */
TEST_F(GF2XTest, DenseMulSmall) {
    /* a(x) = x + 1 = 0b11, b(x) = x + 1 = 0b11
     * a*b = x^2 + 1 = 0b101 (in GF(2), 2x cancels)
     * mod (x^5 - 1): still 0b101 = x^2 + 1 */
    const size_t n = 5;
    const size_t words = GF2X_WORDS(n);
    uint64_t a[1] = {0b11};
    uint64_t b[1] = {0b11};
    uint64_t c[1] = {0};

    gf2x_mul_ref(c, a, b, n);
    /* x^2 + 1 = bit 0 and bit 2 set = 0b101 = 5 */
    EXPECT_EQ(c[0] & 0x1F, 0b00101ULL);
}

TEST_F(GF2XTest, DenseMulIdentity) {
    /* Multiply by 1 (= x^0) should give the same polynomial mod (x^n-1) */
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    std::vector<uint64_t> a(words), one(words, 0), c(words);

    random_poly(a.data(), n, 999);
    one[0] = 1; /* polynomial = 1 */

    gf2x_mul_ref(c.data(), a.data(), one.data(), n);
    EXPECT_TRUE(gf2x_equal(c.data(), a.data(), words))
        << "Multiplication by 1 should return the original polynomial";
}

/* NEON matches Reference */
TEST_F(GF2XTest, DenseMulNeonMatchesRef) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    std::vector<uint64_t> a(words), b(words), c_ref(words), c_neon(words);

    random_poly(a.data(), n, 2026);
    random_poly(b.data(), n, 2027);

    gf2x_mul_ref(c_ref.data(), a.data(), b.data(), n);
    gf2x_mul_neon(c_neon.data(), a.data(), b.data(), n);

    EXPECT_TRUE(gf2x_equal(c_ref.data(), c_neon.data(), words))
        << "NEON dense mul result differs from reference";
}

/* AMX matches Reference */
TEST_F(GF2XTest, DenseMulAmxMatchesRef) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    std::vector<uint64_t> a(words), b(words), c_ref(words), c_amx(words);

    random_poly(a.data(), n, 3030);
    random_poly(b.data(), n, 3031);

    gf2x_mul_ref(c_ref.data(), a.data(), b.data(), n);
    gf2x_mul_amx(c_amx.data(), a.data(), b.data(), n);

    EXPECT_TRUE(gf2x_equal(c_ref.data(), c_amx.data(), words))
        << "AMX dense mul result differs from reference";
}

/* ========================================================================
 * Sparse × Dense Correctness Tests
 * ======================================================================== */

/* Sparse × Dense should match Dense × Dense when sparse poly is expanded */
TEST_F(GF2XTest, SparseMulRefMatchesDense) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    const size_t weight = HQC_128_OMEGA; /* 66 */

    std::vector<uint64_t> a_dense(words), b(words), c_dense(words), c_sparse(words);
    std::vector<uint32_t> positions(weight);

    random_sparse(positions.data(), weight, n, 5555);
    random_poly(b.data(), n, 5556);
    sparse_to_dense(a_dense.data(), positions.data(), weight, n);

    gf2x_mul_ref(c_dense.data(), a_dense.data(), b.data(), n);
    gf2x_mul_sparse_ref(c_sparse.data(), positions.data(), weight, b.data(), n);

    EXPECT_TRUE(gf2x_equal(c_dense.data(), c_sparse.data(), words))
        << "Sparse ref result differs from dense ref result";
}

/* Sparse NEON matches Reference */
TEST_F(GF2XTest, SparseMulNeonMatchesRef) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    const size_t weight = HQC_128_OMEGA;

    std::vector<uint64_t> b(words), c_ref(words), c_neon(words);
    std::vector<uint32_t> positions(weight);

    random_sparse(positions.data(), weight, n, 7777);
    random_poly(b.data(), n, 7778);

    gf2x_mul_sparse_ref(c_ref.data(), positions.data(), weight, b.data(), n);
    gf2x_mul_sparse_neon(c_neon.data(), positions.data(), weight, b.data(), n);

    EXPECT_TRUE(gf2x_equal(c_ref.data(), c_neon.data(), words))
        << "NEON sparse mul result differs from reference";
}

/* Sparse AMX matches Reference */
TEST_F(GF2XTest, SparseMulAmxMatchesRef) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    const size_t weight = HQC_128_OMEGA;

    std::vector<uint64_t> b(words), c_ref(words), c_amx(words);
    std::vector<uint32_t> positions(weight);

    random_sparse(positions.data(), weight, n, 9999);
    random_poly(b.data(), n, 10000);

    gf2x_mul_sparse_ref(c_ref.data(), positions.data(), weight, b.data(), n);
    gf2x_mul_sparse_amx(c_amx.data(), positions.data(), weight, b.data(), n);

    EXPECT_TRUE(gf2x_equal(c_ref.data(), c_amx.data(), words))
        << "AMX sparse mul result differs from reference";
}

/* ========================================================================
 * Commutativity Test
 * ======================================================================== */

TEST_F(GF2XTest, DenseMulCommutativity) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    std::vector<uint64_t> a(words), b(words), ab(words), ba(words);

    random_poly(a.data(), n, 1111);
    random_poly(b.data(), n, 2222);

    gf2x_mul_ref(ab.data(), a.data(), b.data(), n);
    gf2x_mul_ref(ba.data(), b.data(), a.data(), n);

    EXPECT_TRUE(gf2x_equal(ab.data(), ba.data(), words))
        << "Polynomial multiplication should be commutative in F_2[X]/(X^n-1)";
}

/* ========================================================================
 * Performance Benchmarks
 * ======================================================================== */

TEST_F(GF2XTest, BenchmarkDenseMul) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    const int trials = 20;

    std::vector<uint64_t> a(words), b(words), c(words);
    random_poly(a.data(), n, 42);
    random_poly(b.data(), n, 43);

    /* Benchmark Reference C */
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        gf2x_mul_ref(c.data(), a.data(), b.data(), n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                    / (double)trials / 1000.0;

    /* Benchmark NEON PMULL */
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        gf2x_mul_neon(c.data(), a.data(), b.data(), n);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double neon_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                     / (double)trials / 1000.0;

    /* Benchmark AMX */
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        gf2x_mul_amx(c.data(), a.data(), b.data(), n);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double amx_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                    / (double)trials / 1000.0;

    printf("\n");
    printf("========================================================================\n");
    printf("HQC-128 DENSE x DENSE GF(2) POLYNOMIAL MULTIPLICATION (n = %zu)\n", n);
    printf("------------------------------------------------------------------------\n");
    printf("  Reference C (schoolbook) : %10.1f us\n", ref_us);
    printf("  NEON PMULL (clmul)       : %10.1f us\n", neon_us);
    printf("  AMX (int schoolbook)     : %10.1f us\n", amx_us);
    printf("------------------------------------------------------------------------\n");
    printf("  Speedup NEON vs Ref      : %8.2fx\n", ref_us / neon_us);
    printf("  Speedup AMX  vs Ref      : %8.2fx\n", ref_us / amx_us);
    printf("  Speedup NEON vs AMX      : %8.2fx\n", amx_us / neon_us);
    printf("========================================================================\n");
    fflush(stdout);
}

TEST_F(GF2XTest, BenchmarkSparseMul) {
    const size_t n = HQC_128_N;
    const size_t words = GF2X_WORDS(n);
    const size_t weight = HQC_128_OMEGA;
    const int trials = 100;

    std::vector<uint64_t> b(words), c(words);
    std::vector<uint32_t> positions(weight);
    random_sparse(positions.data(), weight, n, 42);
    random_poly(b.data(), n, 43);

    /* Benchmark Reference C */
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        gf2x_mul_sparse_ref(c.data(), positions.data(), weight, b.data(), n);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                    / (double)trials / 1000.0;

    /* Benchmark NEON */
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        gf2x_mul_sparse_neon(c.data(), positions.data(), weight, b.data(), n);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double neon_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                     / (double)trials / 1000.0;

    /* Benchmark AMX */
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; i++) {
        gf2x_mul_sparse_amx(c.data(), positions.data(), weight, b.data(), n);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double amx_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                    / (double)trials / 1000.0;

    printf("\n");
    printf("========================================================================\n");
    printf("HQC-128 SPARSE x DENSE GF(2) POLYNOMIAL MULTIPLICATION\n");
    printf("Parameters: n = %zu, omega = %zu (sparse weight)\n", n, weight);
    printf("------------------------------------------------------------------------\n");
    printf("  Reference C (shift-XOR) : %10.1f us\n", ref_us);
    printf("  NEON (128-bit XOR)      : %10.1f us\n", neon_us);
    printf("  AMX (int accumulate)    : %10.1f us\n", amx_us);
    printf("------------------------------------------------------------------------\n");
    printf("  Speedup NEON vs Ref     : %8.2fx\n", ref_us / neon_us);
    printf("  Speedup AMX  vs Ref     : %8.2fx\n", ref_us / amx_us);
    printf("  Speedup NEON vs AMX     : %8.2fx\n", amx_us / neon_us);
    printf("========================================================================\n");
    fflush(stdout);
}
