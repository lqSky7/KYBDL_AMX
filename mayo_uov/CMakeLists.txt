cmake_minimum_required(VERSION 3.15)
project(mayo_uov_amx C CXX ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/amx
    ${CMAKE_SOURCE_DIR}/neon-ntt/common
)

add_executable(test_mayo_amx
    test_mayo_amx.cpp
    gf16.c
    mayo_amx.c
    ${CMAKE_SOURCE_DIR}/amx/aux_routines.c
    ${CMAKE_SOURCE_DIR}/neon-ntt/common/m1cycles.c
)

target_include_directories(test_mayo_amx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/amx
    ${CMAKE_SOURCE_DIR}/neon-ntt/common
)

target_link_libraries(test_mayo_amx PRIVATE GTest::gtest_main)
gtest_discover_tests(test_mayo_amx DISCOVERY_TIMEOUT 60)
