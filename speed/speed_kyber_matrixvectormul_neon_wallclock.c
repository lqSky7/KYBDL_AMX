/*
 * Wall-clock (mach_absolute_time) variant of speed_kyber_matrixvectormul_neon.c.
 * See wallclock.h for why: this environment doesn't have kpc/kperf access.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "params.h"
#include "indcpa.h"
#include "poly.h"
#include "polyvec.h"
#include "ntt.h"
#include "NTT_params.h"
#include "symmetric.h"
#include "rng.h"
#include "speed.h"

#ifndef NTESTS
#error Please #define NTESTS to the desired number of repetitions for the benchmark
#endif

uint64_t time0, time1;
uint64_t cycles[NTESTS];

#include "wallclock.h"
#define SETUP_COUNTER() \
    {                   \
        (void)cycles;   \
        setup_rdtsc();  \
    }
#define CYCLE_TYPE "%lld"
#define GET_TIME rdtsc()

#undef __AVERAGE__
#define __MEDIAN__

static int cmp_uint64(const void *a, const void *b) {
    return ((*((const uint64_t *)a)) - ((*((const uint64_t *)b))));
}

#define LOOP_INIT(__clock0, __clock1) \
    {}
#define LOOP_TAIL(__f_string, records, __clock0, __clock1)    \
    {                                                         \
        qsort(records, sizeof(uint64_t), NTESTS, cmp_uint64); \
        printf(__f_string, records[NTESTS >> 1]);             \
    }
#define BODY_INIT(__clock0, __clock1) \
    { __clock0 = GET_TIME; }
#define BODY_TAIL(records, __clock0, __clock1) \
    {                                          \
        __clock1 = GET_TIME;                   \
        records[i] = __clock1 - __clock0;      \
    }

#define WRAP_FUNC(__f_string, records, __clock0, __clock1, func) \
    {                                                            \
        LOOP_INIT(__clock0, __clock1);                           \
        for (size_t i = 0; i < NTESTS; i++) {                    \
            BODY_INIT(__clock0, __clock1);                       \
            func;                                                \
            BODY_TAIL(records, __clock0, __clock1);              \
        }                                                        \
        LOOP_TAIL(__f_string, records, __clock0, __clock1);      \
    }

/*
 * NEON NTT-based matrix-vector multiplication.
 * This is the baseline we benchmark against.
 *
 * Full pipeline: gen_matrix -> NTT(s) -> pointwise_mul_montgomery -> INTT
 */
static void MatrixVectorMul_neon(int16_t a[KYBER_K][KYBER_K][KYBER_N],
                                 int16_t sp[KYBER_K][KYBER_N],
                                 int16_t b[KYBER_K][KYBER_N],
                                 const uint8_t seed[KYBER_SYMBYTES]) {
    int16_t sp_asymmetric[KYBER_K][KYBER_N >> 1];

    gen_matrix(a, seed, 0);

    polyvec_ntt(sp);

    for (unsigned int i = 0; i < KYBER_K; i++) {
        KYBER_AARCH64__asm_point_mul_extended(&(sp_asymmetric[i][0]), &(sp[i][0]),
                                              pre_asymmetric_table_extended, asymmetric_const);
    }

    for (unsigned int i = 0; i < KYBER_K; i++) {
        KYBER_AARCH64__asm_asymmetric_mul_montgomery(b[i], &(a[i][0][0]),
                                                     &(sp[0][0]), &(sp_asymmetric[0][0]),
                                                     asymmetric_const);
    }

    polyvec_invntt_to_mont(b);
}

int main() {
    int16_t a[KYBER_K][KYBER_K][KYBER_N];
    int16_t sp[KYBER_K][KYBER_N];
    int16_t b[KYBER_K][KYBER_N];
    uint8_t seed[KYBER_SYMBYTES] = {0};

    unsigned char entropy_input[48] = {0};
    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }
    randombytes_init(entropy_input, NULL, 256);
    randombytes(seed, KYBER_SYMBYTES);

    SETUP_COUNTER();

    WRAP_FUNC("MatrixVectorMul: " CYCLE_TYPE " ns\n", cycles, time0, time1,
              MatrixVectorMul_neon(a, sp, b, seed));

    return 0;
}
