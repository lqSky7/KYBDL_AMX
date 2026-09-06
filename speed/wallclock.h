/*
 * Drop-in replacement for m1cycles.h's setup_rdtsc()/rdtsc() pair, using
 * mach_absolute_time() instead of the private kperf/kpc counters.
 *
 * kpc_set_config() and friends (used by m1cycles.h) require the calling
 * process to run as root; this environment doesn't have that, so we fall
 * back to wall-clock timing in nanoseconds. This is less precise than true
 * retired-cycle counts (subject to DVFS/scheduling noise), but needs no
 * special privileges.
 */
#ifndef WALLCLOCK_H
#define WALLCLOCK_H

#include <mach/mach_time.h>
#include <stdint.h>

static mach_timebase_info_data_t wallclock_tb;

static inline void setup_rdtsc(void) {
    mach_timebase_info(&wallclock_tb);
}

/* Returns elapsed wall-clock time in nanoseconds since an arbitrary epoch. */
static inline uint64_t rdtsc(void) {
    uint64_t t = mach_absolute_time();
    return (uint64_t)((__uint128_t)t * wallclock_tb.numer / wallclock_tb.denom);
}

#endif /* WALLCLOCK_H */
