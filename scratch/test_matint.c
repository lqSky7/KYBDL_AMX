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
    AMX_CLR();

    int8_t x[64] __attribute__((aligned(128)));
    int8_t y[64] __attribute__((aligned(128)));
    int32_t z_out[32 * 32] __attribute__((aligned(128)));

    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));
    memset(z_out, 0, sizeof(z_out));

    for (int i = 0; i < 32; i++) {
        x[i] = 10;
        y[i] = 10;
    }

    AMX_LDX(AMX_PTR(x) | LDX_REG(0));
    AMX_LDY(AMX_PTR(y) | LDY_REG(0));

    /* MATINT: 8-bit integer matrix multiply-accumulate */
    AMX_MATINT(MATINT_ALU_MODE_Z_ADD_X_MUL_Y | MATINT_X_REG(0) | MATINT_Y_REG(0) | MATINT_Z_ROW(0));

    for (int r = 0; r < 32; r++) {
        AMX_STZ(AMX_PTR(&z_out[16 * r]) | STZ_Z_ROW(r));
    }

    printf("MATINT: z[0] = %d (expected 100?)\n", z_out[0]);
    for (int r = 0; r < 4; r++) {
        printf("Row %d: %d %d %d %d\n", r, z_out[16 * r], z_out[16 * r + 1], z_out[16 * r + 2], z_out[16 * r + 3]);
    }

    AMX_CLR();
    return 0;
}
