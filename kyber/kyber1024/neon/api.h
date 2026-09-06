#ifndef KYBER_AARCH64_API_H
#define KYBER_AARCH64_API_H

#include "params.h"
#include <stdint.h>

#define CRYPTO_SECRETKEYBYTES  KYBER_SECRETKEYBYTES
#define CRYPTO_PUBLICKEYBYTES  KYBER_PUBLICKEYBYTES
#define CRYPTO_CIPHERTEXTBYTES KYBER_CIPHERTEXTBYTES
#define CRYPTO_BYTES           KYBER_SSBYTES
#define CRYPTO_ALGNAME         "Kyber1024"

#define crypto_kem_keypair KYBER_NAMESPACE(crypto_kem_keypair)
#define crypto_kem_enc     KYBER_NAMESPACE(crypto_kem_enc)
#define crypto_kem_dec     KYBER_NAMESPACE(crypto_kem_dec)

#ifdef __cplusplus
extern "C" {
#endif

int crypto_kem_keypair(uint8_t *pk, uint8_t *sk);
int crypto_kem_enc(uint8_t *ct, uint8_t *ss, const uint8_t *pk);
int crypto_kem_dec(uint8_t *ss, const uint8_t *ct, const uint8_t *sk);

#ifdef __cplusplus
}
#endif

#endif
