# HQC GF(2) Polynomial Multiplication — Build Configuration
#
# Three implementations:
#   1. Reference C (schoolbook)
#   2. NEON PMULL (carry-less multiply)
#   3. AMX (integer-encoded outer products)

# Common sources (shared utility functions live in gf2x_ref.c)
set(HQC_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gf2x_ref.c
)

# NEON PMULL sources
set(HQC_NEON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gf2x_neon.c
)

# AMX sources
set(HQC_AMX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/amx/hqc_amx_gf2x.c
)

# Include paths
set(HQC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amx
    ${AMX_PATH}
    ${CMAKE_SOURCE_DIR}/neon-ntt/common
)

# ---- GF(2) Polynomial Multiplication Test (3-way correctness + benchmark) ----
add_executable(test_hqc_gf2x
    ${CMAKE_SOURCE_DIR}/test/test_hqc_gf2x.cpp
    ${HQC_COMMON_SOURCES}
    ${HQC_NEON_SOURCES}
    ${HQC_AMX_SOURCES}
)

target_include_directories(test_hqc_gf2x PRIVATE ${HQC_INCLUDE_DIRS})
target_compile_options(test_hqc_gf2x PRIVATE -O3)
target_link_libraries(test_hqc_gf2x PRIVATE GTest::gtest_main)
gtest_discover_tests(test_hqc_gf2x DISCOVERY_TIMEOUT ${GTEST_DISCOVERY_TIMEOUT})
