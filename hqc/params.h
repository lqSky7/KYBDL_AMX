/*
 * HQC Parameter Definitions
 *
 * Hamming Quasi-Cyclic KEM — NIST Post-Quantum Standard (selected March 2025)
 * Ring: F_2[X] / (X^n - 1), n prime
 *
 * Reference: https://pqc-hqc.org/
 */

#ifndef HQC_PARAMS_H
#define HQC_PARAMS_H

#include <stdint.h>
#include <stddef.h>

/*
 * Utility macros for GF(2) polynomial storage
 * Polynomials are stored as packed bit arrays in uint64_t words.
 * a[i] stores bits [64*i, 64*i+63] of the polynomial.
 */
#define GF2X_WORDS(n) (((n) + 63) / 64)
#define GF2X_BYTES(n) (GF2X_WORDS(n) * sizeof(uint64_t))

/* ========================================================================
 * HQC-128 (NIST Security Level 1, equivalent to AES-128)
 * ======================================================================== */
#define HQC_128_N           17669   /* Polynomial ring dimension (prime) */
#define HQC_128_N_WORDS     GF2X_WORDS(HQC_128_N)  /* = 277 */
#define HQC_128_N_BYTES     (HQC_128_N_WORDS * 8)  /* = 2216 */
#define HQC_128_OMEGA       66      /* Secret key weight (Hamming weight of x, y) */
#define HQC_128_OMEGA_R     75      /* Randomness weight (weight of r1, r2) */
#define HQC_128_OMEGA_E     75      /* Error weight (weight of e) */

/* Reed-Solomon parameters for HQC-128 */
#define HQC_128_RS_N1       46      /* RS codeword length */
#define HQC_128_RS_K        16      /* RS message length */
#define HQC_128_RS_D        15      /* RS minimum distance */

/* Reed-Muller parameters for HQC-128 */
#define HQC_128_RM_N2       384     /* RM codeword length */
#define HQC_128_RM_M        8       /* RM order parameter */
#define HQC_128_RM_D        192     /* RM minimum distance */

#define HQC_128_RM_MULT     3       /* Duplication multiplicity */

/* ========================================================================
 * HQC-192 (NIST Security Level 3, equivalent to AES-192)
 * ======================================================================== */
#define HQC_192_N           35851
#define HQC_192_N_WORDS     GF2X_WORDS(HQC_192_N)  /* = 561 */
#define HQC_192_N_BYTES     (HQC_192_N_WORDS * 8)
#define HQC_192_OMEGA       100
#define HQC_192_OMEGA_R     114
#define HQC_192_OMEGA_E     114

#define HQC_192_RS_N1       56
#define HQC_192_RS_K        24
#define HQC_192_RS_D        16

#define HQC_192_RM_N2       640
#define HQC_192_RM_M        8
#define HQC_192_RM_D        320

#define HQC_192_RM_MULT     5

/* ========================================================================
 * HQC-256 (NIST Security Level 5, equivalent to AES-256)
 * ======================================================================== */
#define HQC_256_N           57637
#define HQC_256_N_WORDS     GF2X_WORDS(HQC_256_N)  /* = 901 */
#define HQC_256_N_BYTES     (HQC_256_N_WORDS * 8)
#define HQC_256_OMEGA       131
#define HQC_256_OMEGA_R     149
#define HQC_256_OMEGA_E     149

#define HQC_256_RS_N1       90
#define HQC_256_RS_K        32
#define HQC_256_RS_D        29

#define HQC_256_RM_N2       640
#define HQC_256_RM_M        8
#define HQC_256_RM_D        320

#define HQC_256_RM_MULT     5

/* ========================================================================
 * Active parameter set selection (default: HQC-128)
 * Override at compile time with -DHQC_PARAM_SET=192 or =256
 * ======================================================================== */
#ifndef HQC_PARAM_SET
#define HQC_PARAM_SET 128
#endif

#if HQC_PARAM_SET == 128
    #define HQC_N           HQC_128_N
    #define HQC_N_WORDS     HQC_128_N_WORDS
    #define HQC_N_BYTES     HQC_128_N_BYTES
    #define HQC_OMEGA       HQC_128_OMEGA
    #define HQC_OMEGA_R     HQC_128_OMEGA_R
    #define HQC_OMEGA_E     HQC_128_OMEGA_E
#elif HQC_PARAM_SET == 192
    #define HQC_N           HQC_192_N
    #define HQC_N_WORDS     HQC_192_N_WORDS
    #define HQC_N_BYTES     HQC_192_N_BYTES
    #define HQC_OMEGA       HQC_192_OMEGA
    #define HQC_OMEGA_R     HQC_192_OMEGA_R
    #define HQC_OMEGA_E     HQC_192_OMEGA_E
#elif HQC_PARAM_SET == 256
    #define HQC_N           HQC_256_N
    #define HQC_N_WORDS     HQC_256_N_WORDS
    #define HQC_N_BYTES     HQC_256_N_BYTES
    #define HQC_OMEGA       HQC_256_OMEGA
    #define HQC_OMEGA_R     HQC_256_OMEGA_R
    #define HQC_OMEGA_E     HQC_256_OMEGA_E
#else
    #error "Invalid HQC_PARAM_SET. Must be 128, 192, or 256."
#endif

#endif /* HQC_PARAMS_H */
