#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include "aarch64.h"
#include "amx.h"

int main(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    AMX_SET();

    int16_t x[32] __attribute__((aligned(128)));
    int16_t y[32] __attribute__((aligned(128)));
    int16_t z_out[32 * 32] __attribute__((aligned(128)));

    for (int i = 0; i < 32; i++) {
        x[i] = 100;
        y[i] = 100;
    }

    AMX_LDX(AMX_PTR(x) | LDX_REG(0));
    AMX_LDY(AMX_PTR(y) | LDY_REG(0));

    /* 100 * 100 = 10,000. After 4 iterations = 40,000 > 32767 */
    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);
    for (int iter = 1; iter < 4; iter++) {
        AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0));
    }

    for (int r = 0; r < 32; r++) {
        AMX_STZ(AMX_PTR(&z_out[32 * r]) | STZ_Z_ROW(2 * r));
    }

    printf("4x (100*100) = %d (signed 16-bit: %d, uint16: %u, expected 40000)\n", 
           z_out[0], (int16_t)z_out[0], (uint16_t)z_out[0]);

    AMX_CLR();
    return 0;
}
