/*
 * Streamlined NTRU Prime — sntrup761 Parameters
 *
 * Ring: Z_q[x] / (x^p - x - 1)
 * Reference: https://ntruprime.cr.yp.to/
 */

#ifndef SNTRUP761_PARAMS_H
#define SNTRUP761_PARAMS_H

#define SNTRUP_P       761     /* Polynomial degree (prime) */
#define SNTRUP_Q       4591    /* Coefficient modulus (prime) */
#define SNTRUP_Q_HALF  2295    /* (q-1)/2, for centered representation */

/* AMX block parameters */
#define SNTRUP_BLOCK   32      /* Coefficients per AMX register (32 x int16) */
#define SNTRUP_NBLOCKS 24      /* ceil(761/32) = 24 blocks */
#define SNTRUP_PADDED  768     /* 24 * 32 = 768 (zero-padded length) */

/* Maximum safe accumulation depth for ternary x general multiply.
 * Each product |r[j]*h[k]| <= 4590. For signed 16-bit: k*4590 < 32768 => k <= 7 */
#define SNTRUP_MAX_ACCUM 7

#endif /* SNTRUP761_PARAMS_H */
