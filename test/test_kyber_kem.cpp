/*
 * Test for AMX Kyber KEM correctness.
 *
 * Validates that encrypt(decrypt(m)) = m for the AMX variant.
 * This does NOT compare against NEON KAT vectors (the AMX variant stores
 * keys in coefficient domain while NEON stores them in NTT domain, so
 * the serialized formats differ).
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

extern "C" {
#include "api.h"
#include "randombytes.h"
}

#ifndef TEST_NAME
#define TEST_NAME kyber_amx_test
#endif

#define XSTR(s) STR(s)
#define STR(s) #s

class KyberAMXKEMTest : public ::testing::Test {};

TEST(KyberAMXKEMTest, KeygenEncDecRoundtrip) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
    uint8_t ss_enc[CRYPTO_BYTES];
    uint8_t ss_dec[CRYPTO_BYTES];
    uint8_t entropy_input[48];

    for (int i = 0; i < 48; i++)
        entropy_input[i] = i;
    randombytes_init(entropy_input, NULL, 256);

    for (int trial = 0; trial < 10; trial++) {
        int rc;

        rc = crypto_kem_keypair(pk, sk);
        ASSERT_EQ(rc, 0) << "keypair failed at trial " << trial;

        rc = crypto_kem_enc(ct, ss_enc, pk);
        ASSERT_EQ(rc, 0) << "enc failed at trial " << trial;

        rc = crypto_kem_dec(ss_dec, ct, sk);
        ASSERT_EQ(rc, 0) << "dec failed at trial " << trial;

        ASSERT_EQ(memcmp(ss_enc, ss_dec, CRYPTO_BYTES), 0)
            << "Shared secret mismatch at trial " << trial;
    }
}

TEST(KyberAMXKEMTest, DecapsulationOfModifiedCiphertext) {
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
    uint8_t ss_enc[CRYPTO_BYTES];
    uint8_t ss_dec[CRYPTO_BYTES];
    uint8_t entropy_input[48];

    for (int i = 0; i < 48; i++)
        entropy_input[i] = i + 100;
    randombytes_init(entropy_input, NULL, 256);

    crypto_kem_keypair(pk, sk);
    crypto_kem_enc(ct, ss_enc, pk);

    /* Flip a bit in the ciphertext */
    ct[0] ^= 1;

    crypto_kem_dec(ss_dec, ct, sk);

    /* Modified ciphertext should produce different shared secret (CCA security) */
    ASSERT_NE(memcmp(ss_enc, ss_dec, CRYPTO_BYTES), 0)
        << "Modified ciphertext should not produce the same shared secret";
}
