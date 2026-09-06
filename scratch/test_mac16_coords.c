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

    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));
    memset(z_out, 0, sizeof(z_out));

    x[0] = 2; x[1] = 3;
    y[0] = 5; y[1] = 7;

    AMX_LDX(AMX_PTR(x) | LDX_REG(0));
    AMX_LDY(AMX_PTR(y) | LDY_REG(0));

    AMX_MAC16(MAC16_MATRIX | MAC16_X_REG(0) | MAC16_Y_REG(0) | MAC16_Z_ROW(0) | MAC16_Z_SKIP);

    for (int r = 0; r < 32; r++) {
        AMX_STZ(AMX_PTR(&z_out[32 * r]) | STZ_Z_ROW(2 * r));
    }

    printf("z[0][0] (x0*y0 = 2*5): %d (expected 10)\n", z_out[0 * 32 + 0]);
    printf("z[0][1] (x0*y1 = 2*7): %d (if X is row, Y is col -> 14. If Y is row, X is col -> 15)\n", z_out[0 * 32 + 1]);
    printf("z[1][0] (x1*y0 = 3*5): %d (if X is row, Y is col -> 15. If Y is row, X is col -> 14)\n", z_out[1 * 32 + 0]);
    printf("z[1][1] (x1*y1 = 3*7): %d (expected 21)\n", z_out[1 * 32 + 1]);

    AMX_CLR();
    return 0;
}
