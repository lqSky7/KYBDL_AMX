/*
 * Wall-clock (mach_absolute_time) variant of speed_kyber_matrixvectormul_amx.c.
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
#include "symmetric.h"
#include "rng.h"
#include "speed.h"
#include "kyber_amx_polymul.h"
#include "aarch64.h"

#ifdef ALLOC_MMAP
#include "memory_alloc.h"
#endif

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
        fflush(stdout);                                       \
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
 * AMX initialization is handled automatically by the __attribute__((
 * constructor)) in kyber/amx/kyber_amx_indcpa.c, which every AMX Kyber
 * library links. Do NOT call AMX_SET() again here: on this hardware,
 * calling AMX_SET() a second time without an intervening AMX_CLR() hangs
 * indefinitely (confirmed with a minimal repro outside this codebase).
 */

int main() {
    int16_t a[KYBER_K][KYBER_K][KYBER_N];
    int16_t s[KYBER_K][KYBER_N];
    int16_t res[KYBER_K][KYBER_N];
    uint8_t seed[KYBER_SYMBYTES] = {0};

    unsigned char entropy_input[48] = {0};
    for (int i = 0; i < 48; i++) {
        entropy_input[i] = i;
    }
    randombytes_init(entropy_input, NULL, 256);
    randombytes(seed, KYBER_SYMBYTES);

    /* Generate random secret vector */
    for (int i = 0; i < KYBER_K; i++) {
        for (int j = 0; j < KYBER_N; j++) {
            s[i][j] = rand() % KYBER_Q;
        }
    }

    SETUP_COUNTER();

    /* Benchmark: gen_matrix + AMX matrix-vector multiply */
    WRAP_FUNC("MatrixVectorMul: " CYCLE_TYPE " ns\n", cycles, time0, time1, {
        gen_matrix(a, seed, 0);
        amx_kyber_matrix_vector_mul(&(res[0][0]), &(a[0][0][0]), &(s[0][0]), KYBER_K);
    });

    /* Benchmark: AMX matrix-vector multiply only (without gen_matrix) */
    gen_matrix(a, seed, 0);
    WRAP_FUNC("MatrixVectorMul (no gen_matrix): " CYCLE_TYPE " ns\n", cycles, time0, time1,
              amx_kyber_matrix_vector_mul(&(res[0][0]), &(a[0][0][0]), &(s[0][0]), KYBER_K));

    return 0;
}
