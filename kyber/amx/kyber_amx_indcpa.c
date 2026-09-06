/*
 * AMX variant of Kyber IND-CPA encryption.
 *
 * This replaces the NTT-based polynomial multiplication with AMX TMVP.
 * The key difference: instead of working in the NTT domain, we stay in
 * coefficient domain and use amx_kyber_poly_basemul for multiplication.
 *
 * License: CC0 1.0 / MIT (matching the original)
 */

#include "params.h"
#include "rejsample.h"
#include "indcpa.h"
#include "poly.h"
#include "polyvec.h"
#include "randombytes.h"
#include "symmetric.h"
#include "kyber_amx_polymul.h"
#include "aarch64.h"
#include "amx.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * AMX must be explicitly enabled once per process before any AMX/MAC16
 * instruction executes (see amx/frodokem_opt_matmul.c and the Saber
 * AMX poly.c files under neon-ntt for the same pattern). This function
 * was previously defined but never invoked, meaning every AMX
 * instruction in the Kyber AMX KEM path executed without AMX enabled.
 */
__attribute__((constructor)) static void ensure_amx_init(void) {
    AMX_SET();
}

/* Re-use pack/unpack from the NEON implementation — these are data format only */
static void pack_pk(uint8_t r[KYBER_INDCPA_PUBLICKEYBYTES],
                    int16_t pk[KYBER_K][KYBER_N],
                    const uint8_t seed[KYBER_SYMBYTES]) {
    polyvec_tobytes(r, pk);
    memcpy(r + KYBER_POLYVECBYTES, seed, KYBER_SYMBYTES);
}

static void unpack_pk(int16_t pk[KYBER_K][KYBER_N],
                      uint8_t seed[KYBER_SYMBYTES],
                      const uint8_t packedpk[KYBER_INDCPA_PUBLICKEYBYTES]) {
    polyvec_frombytes(pk, packedpk);
    memcpy(seed, packedpk + KYBER_POLYVECBYTES, KYBER_SYMBYTES);
}

static void pack_sk(uint8_t r[KYBER_INDCPA_SECRETKEYBYTES],
                    int16_t sk[KYBER_K][KYBER_N]) {
    polyvec_tobytes(r, sk);
}

static void unpack_sk(int16_t sk[KYBER_K][KYBER_N],
                      const uint8_t packedsk[KYBER_INDCPA_SECRETKEYBYTES]) {
    polyvec_frombytes(sk, packedsk);
}

static void pack_ciphertext(uint8_t r[KYBER_INDCPA_BYTES],
                            int16_t b[KYBER_K][KYBER_N], int16_t *v) {
    polyvec_compress(r, b);
    poly_compress(r + KYBER_POLYVECCOMPRESSEDBYTES, v);
}

static void unpack_ciphertext(int16_t b[KYBER_K][KYBER_N], int16_t *v,
                              const uint8_t c[KYBER_INDCPA_BYTES]) {
    polyvec_decompress(b, c);
    poly_decompress(v, c + KYBER_POLYVECCOMPRESSEDBYTES);
}

#define gen_a(A,B)  gen_matrix(A,B,0)
#define gen_at(A,B) gen_matrix(A,B,1)

/*
 * gen_matrix: unchanged from NEON version.
 * Generates the public matrix A from seed. Output is in coefficient domain.
 * In the NEON implementation, A is generated and then used in NTT domain.
 * Here, we keep it in coefficient domain for TMVP.
 *
 * NOTE: The gen_matrix function from the NEON code produces coefficients
 * already in [0, q), which is exactly what our AMX TMVP needs.
 */
#define GEN_MATRIX_NBLOCKS ((12*KYBER_N/8*(1 << 12)/KYBER_Q + XOF_BLOCKBYTES)/XOF_BLOCKBYTES)

void gen_matrix(int16_t a[KYBER_K][KYBER_K][KYBER_N],
                const uint8_t seed[KYBER_SYMBYTES], int transposed) {
    unsigned int ctr0, ctr1, k;
    unsigned int buflen, off;
    uint8_t buf0[GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES + 2],
            buf1[GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES + 2];
    neon_xof_state state;

    #if KYBER_K == 2
    for (unsigned int i = 0; i < KYBER_K; i++) {
        if (transposed)
            xofx2_absorb(&state, seed, i, i, 0, 1);
        else
            xofx2_absorb(&state, seed, 0, 1, i, i);

        xofx2_squeezeblocks(buf0, buf1, GEN_MATRIX_NBLOCKS, &state);
        buflen = GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES;
        ctr0 = neon_rej_uniform(&(a[i][0][0]), buf0);
        ctr1 = neon_rej_uniform(&(a[i][1][0]), buf1);
        while (ctr0 < KYBER_N || ctr1 < KYBER_N) {
            off = buflen % 3;
            for (k = 0; k < off; k++) {
                buf0[k] = buf0[buflen - off + k];
                buf1[k] = buf1[buflen - off + k];
            }
            xofx2_squeezeblocks(buf0 + off, buf1 + off, 1, &state);
            buflen = off + XOF_BLOCKBYTES;
            ctr0 += rej_uniform(&(a[i][0][0]) + ctr0, KYBER_N - ctr0, buf0, buflen);
            ctr1 += rej_uniform(&(a[i][1][0]) + ctr1, KYBER_N - ctr1, buf1, buflen);
        }
    }
    #elif KYBER_K == 3
    int16_t *s1 = NULL, *s2 = NULL;
    unsigned int x1, x2, y1, y2;
    xof_state c_state;

    for (unsigned int j = 0; j < KYBER_K * KYBER_K - 1; j += 2) {
        switch (j) {
        case 0: s1=&(a[0][0][0]); s2=&(a[0][1][0]); x1=0; y1=0; x2=0; y2=1; break;
        case 2: s1=&(a[0][2][0]); s2=&(a[1][0][0]); x1=0; y1=2; x2=1; y2=0; break;
        case 4: s1=&(a[1][1][0]); s2=&(a[1][2][0]); x1=1; y1=1; x2=1; y2=2; break;
        default: s1=&(a[2][0][0]); s2=&(a[2][1][0]); x1=2; y1=0; x2=2; y2=1; break;
        }

        if (transposed)
            xofx2_absorb(&state, seed, x1, x2, y1, y2);
        else
            xofx2_absorb(&state, seed, y1, y2, x1, x2);

        xofx2_squeezeblocks(buf0, buf1, GEN_MATRIX_NBLOCKS, &state);
        buflen = GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES;
        ctr0 = neon_rej_uniform(s1, buf0);
        ctr1 = neon_rej_uniform(s2, buf1);

        while (ctr0 < KYBER_N || ctr1 < KYBER_N) {
            off = buflen % 3;
            for (k = 0; k < off; k++) {
                buf0[k] = buf0[buflen - off + k];
                buf1[k] = buf1[buflen - off + k];
            }
            xofx2_squeezeblocks(buf0 + off, buf1 + off, 1, &state);
            buflen = off + XOF_BLOCKBYTES;
            ctr0 += rej_uniform(s1 + ctr0, KYBER_N - ctr0, buf0, buflen);
            ctr1 += rej_uniform(s2 + ctr1, KYBER_N - ctr1, buf1, buflen);
        }
    }
    /* Last entry [2][2] */
    if (transposed)
        xof_absorb(&c_state, seed, 2, 2);
    else
        xof_absorb(&c_state, seed, 2, 2);

    xof_squeezeblocks(buf0, GEN_MATRIX_NBLOCKS, &c_state);
    buflen = GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES;
    ctr0 = neon_rej_uniform(&(a[2][2][0]), buf0);
    while (ctr0 < KYBER_N) {
        off = buflen % 3;
        for (k = 0; k < off; k++)
            buf0[k] = buf0[buflen - off + k];
        xof_squeezeblocks(buf0 + off, 1, &c_state);
        buflen = off + XOF_BLOCKBYTES;
        ctr0 += rej_uniform(&(a[2][2][0]) + ctr0, KYBER_N - ctr0, buf0, buflen);
    }
    #elif KYBER_K == 4
    for (unsigned int i = 0; i < KYBER_K; i++) {
        for (unsigned int j = 0; j < KYBER_K; j += 2) {
            if (transposed)
                xofx2_absorb(&state, seed, i, i, j, j + 1);
            else
                xofx2_absorb(&state, seed, j, j + 1, i, i);

            xofx2_squeezeblocks(buf0, buf1, GEN_MATRIX_NBLOCKS, &state);
            buflen = GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES;
            ctr0 = neon_rej_uniform(&(a[i][j][0]), buf0);
            ctr1 = neon_rej_uniform(&(a[i][j + 1][0]), buf1);
            while (ctr0 < KYBER_N || ctr1 < KYBER_N) {
                off = buflen % 3;
                for (k = 0; k < off; k++) {
                    buf0[k] = buf0[buflen - off + k];
                    buf1[k] = buf1[buflen - off + k];
                }
                xofx2_squeezeblocks(buf0 + off, buf1 + off, 1, &state);
                buflen = off + XOF_BLOCKBYTES;
                ctr0 += rej_uniform(&(a[i][j][0]) + ctr0, KYBER_N - ctr0, buf0, buflen);
                ctr1 += rej_uniform(&(a[i][j + 1][0]) + ctr1, KYBER_N - ctr1, buf1, buflen);
            }
        }
    }
    #else
    #error "KYBER_K must be in {2,3,4}"
    #endif
}

/*
 * AMX Key Generation
 *
 * NEON version:
 *   gen_a(a) -> polyvec_ntt(s) -> pointwise_mul(a_hat, s_hat) -> add(e)
 *   sk = s_hat (NTT domain), pk = a*s + e (NTT domain)
 *
 * AMX version:
 *   gen_a(a) -> amx_matrix_vector_mul(a, s) -> add(e)
 *   sk = s (coefficient domain), pk = a*s + e (coefficient domain)
 *
 * IMPORTANT: The NEON version stores sk in NTT domain and pk in NTT domain.
 * Our AMX version stores sk and pk in coefficient domain. This means the
 * pack/unpack functions work the same (they serialize coefficient arrays),
 * but KAT vectors will differ from the NEON version. We ensure correctness
 * by verifying enc(dec(m)) = m for all messages.
 */
void indcpa_keypair_derand(uint8_t pk[KYBER_INDCPA_PUBLICKEYBYTES],
                           uint8_t sk[KYBER_INDCPA_SECRETKEYBYTES],
                           const uint8_t coins[KYBER_SYMBYTES]) {
    uint8_t buf[2 * KYBER_SYMBYTES];
    const uint8_t *publicseed = buf;
    const uint8_t *noiseseed = buf + KYBER_SYMBYTES;
    int16_t a[KYBER_K][KYBER_K][KYBER_N];
    int16_t e[KYBER_K][KYBER_N];
    int16_t pkpv[KYBER_K][KYBER_N];
    int16_t skpv[KYBER_K][KYBER_N];

    memcpy(buf, coins, KYBER_SYMBYTES);
    buf[KYBER_SYMBYTES] = KYBER_K;
    hash_g(buf, buf, KYBER_SYMBYTES + 1);

    gen_a(a, publicseed);

    /* Generate noise polynomials s and e */
#if KYBER_K == 2
    poly_getnoise_eta1_x2(&(skpv[0][0]), &(skpv[1][0]), noiseseed, 0, 1);
    poly_getnoise_eta1_x2(&(e[0][0]), &(e[1][0]), noiseseed, 2, 3);
#elif KYBER_K == 3
    poly_getnoise_eta1_x2(&(skpv[0][0]), &(skpv[1][0]), noiseseed, 0, 1);
    poly_getnoise_eta1_x2(&(skpv[2][0]), &(e[0][0]), noiseseed, 2, 3);
    poly_getnoise_eta1_x2(&(e[1][0]), &(e[2][0]), noiseseed, 4, 5);
#elif KYBER_K == 4
    poly_getnoise_eta1_x2(&(skpv[0][0]), &(skpv[1][0]), noiseseed, 0, 1);
    poly_getnoise_eta1_x2(&(skpv[2][0]), &(skpv[3][0]), noiseseed, 2, 3);
    poly_getnoise_eta1_x2(&(e[0][0]), &(e[1][0]), noiseseed, 4, 5);
    poly_getnoise_eta1_x2(&(e[2][0]), &(e[3][0]), noiseseed, 6, 7);
#endif

    /*
     * AMX: Compute pkpv = A * s using TMVP (coefficient domain)
     * No NTT forward/inverse needed!
     */
    amx_kyber_matrix_vector_mul(&(pkpv[0][0]),
                                &(a[0][0][0]),
                                &(skpv[0][0]),
                                KYBER_K);

    /* pkpv += e, with reduction */
    for (int i = 0; i < KYBER_K; i++) {
        amx_kyber_poly_add(pkpv[i], pkpv[i], e[i]);
    }

    /* Reduce all coefficients to canonical form [0, q) */
    for (int i = 0; i < KYBER_K; i++) {
        poly_reduce(pkpv[i]);
    }

    pack_sk(sk, skpv);
    pack_pk(pk, pkpv, publicseed);
}

/*
 * AMX Encryption
 *
 * NEON: unpack pk -> gen_at -> ntt(r) -> pointwise_mul -> invntt -> add noise
 * AMX: unpack pk -> gen_at -> amx_tmvp(A^T, r) -> add noise
 */
void indcpa_enc(uint8_t c[KYBER_INDCPA_BYTES],
                const uint8_t m[KYBER_INDCPA_MSGBYTES],
                const uint8_t pk[KYBER_INDCPA_PUBLICKEYBYTES],
                const uint8_t coins[KYBER_SYMBYTES]) {
    uint8_t seed[KYBER_SYMBYTES];
    int16_t at[KYBER_K][KYBER_K][KYBER_N];
    int16_t sp[KYBER_K][KYBER_N];
    int16_t pkpv[KYBER_K][KYBER_N];
    int16_t ep[KYBER_K][KYBER_N];
    int16_t b[KYBER_K][KYBER_N];
    int16_t v[KYBER_N];
    int16_t k_msg[KYBER_N];
    int16_t epp[KYBER_N];

    unpack_pk(pkpv, seed, pk);
    poly_frommsg(k_msg, m);
    gen_at(at, seed);

    /* Generate noise polynomials */
#if KYBER_K == 2
    poly_getnoise_eta1_x2(&(sp[0][0]), &(sp[1][0]), coins, 0, 1);
    poly_getnoise_eta2_x2(&(ep[0][0]), &(ep[1][0]), coins, 2, 3);
    poly_getnoise_eta2(&(epp[0]), coins, 4);
#elif KYBER_K == 3
#if KYBER_ETA1 == KYBER_ETA2
    poly_getnoise_eta1_x2(&(sp[0][0]), &(sp[1][0]), coins, 0, 1);
    poly_getnoise_eta1_x2(&(sp[2][0]), &(ep[0][0]), coins, 2, 3);
    poly_getnoise_eta1_x2(&(ep[1][0]), &(ep[2][0]), coins, 4, 5);
    poly_getnoise_eta2(&(epp[0]), coins, 6);
#else
#error "We need eta1 == eta2 here"
#endif
#elif KYBER_K == 4
#if KYBER_ETA1 == KYBER_ETA2
    poly_getnoise_eta1_x2(&(sp[0][0]), &(sp[1][0]), coins, 0, 1);
    poly_getnoise_eta1_x2(&(sp[2][0]), &(sp[3][0]), coins, 2, 3);
    poly_getnoise_eta1_x2(&(ep[0][0]), &(ep[1][0]), coins, 4, 5);
    poly_getnoise_eta1_x2(&(ep[2][0]), &(ep[3][0]), coins, 6, 7);
    poly_getnoise_eta2(&(epp[0]), coins, 8);
#else
#error "We need eta1 == eta2 here"
#endif
#endif

    /*
     * AMX: b = A^T * r using TMVP (coefficient domain)
     */
    amx_kyber_matrix_vector_mul(&(b[0][0]),
                                &(at[0][0][0]),
                                &(sp[0][0]),
                                KYBER_K);

    /* v = pk^T * r (inner product) */
    amx_kyber_inner_prod(v, &(pkpv[0][0]), &(sp[0][0]), KYBER_K);

    /* b += ep */
    for (int i = 0; i < KYBER_K; i++) {
        amx_kyber_poly_add(b[i], b[i], ep[i]);
        poly_reduce(b[i]);
    }

    /* v += epp + m */
    amx_kyber_poly_add(v, v, epp);
    amx_kyber_poly_add(v, v, k_msg);
    poly_reduce(v);

    pack_ciphertext(c, b, v);
}

/*
 * AMX Decryption
 *
 * NEON: unpack ct -> ntt(b) -> pointwise_mul(sk, b) -> invntt -> v - mp
 * AMX: unpack ct -> amx_tmvp(s, b) -> v - mp
 */
void indcpa_dec(uint8_t m[KYBER_INDCPA_MSGBYTES],
                const uint8_t c[KYBER_INDCPA_BYTES],
                const uint8_t sk[KYBER_INDCPA_SECRETKEYBYTES]) {
    int16_t b[KYBER_K][KYBER_N];
    int16_t skpv[KYBER_K][KYBER_N];
    int16_t v[KYBER_N];
    int16_t mp[KYBER_N];

    unpack_ciphertext(b, v, c);
    unpack_sk(skpv, sk);

    /* AMX: mp = s^T * b (inner product in coefficient domain) */
    amx_kyber_inner_prod(mp, &(skpv[0][0]), &(b[0][0]), KYBER_K);

    /* v = v - mp */
    amx_kyber_poly_sub(v, v, mp);
    poly_reduce(v);

    poly_tomsg(m, v);
}
