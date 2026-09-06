# sntrup761 Polynomial Multiplication — Build Configuration
#
# Three implementations:
#   1. Reference C (schoolbook, 32-bit accumulator)
#   2. NEON vmlal (widening multiply-accumulate)
#   3. AMX mac16 (outer product blocks)

set(SNTRUP_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_ref.c
)

set(SNTRUP_NEON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_neon.c
)

set(SNTRUP_AMX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/amx/sntrup_amx_polymul.c
)

set(SNTRUP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amx
    ${AMX_PATH}
    ${CMAKE_SOURCE_DIR}/neon-ntt/common
)

# ---- Polynomial Multiplication Test (3-way correctness + benchmark) ----
add_executable(test_sntrup_polymul
    ${CMAKE_SOURCE_DIR}/test/test_sntrup_polymul.cpp
    ${SNTRUP_COMMON_SOURCES}
    ${SNTRUP_NEON_SOURCES}
    ${SNTRUP_AMX_SOURCES}
)

target_include_directories(test_sntrup_polymul PRIVATE
    ${SNTRUP_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
)
target_compile_options(test_sntrup_polymul PRIVATE -O3)
target_link_libraries(test_sntrup_polymul PRIVATE GTest::gtest_main)
gtest_discover_tests(test_sntrup_polymul DISCOVERY_TIMEOUT ${GTEST_DISCOVERY_TIMEOUT})
