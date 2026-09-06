set(KYBER_PARAMETER_SETS kyber512 kyber768 kyber1024)

# Common sources shared between neon and amx
set(KYBER_SOURCES_COMMON
    kem.c poly.c polyvec.c reduce.c rejsample.c symmetric-shake.c verify.c cbd.c ntt.c
    __asm_poly.S __asm_NTT.S __asm_iNTT.S __asm_base_mul.S
)

set(KYBER_SOURCES_neon indcpa.c)
set(KYBER_SOURCES_amx)

# AMX-specific sources
# Note: we also include aux_routines from the amx root
set(AMX_SOURCES_amx
    ${CMAKE_SOURCE_DIR}/amx/aux_routines.c
    ${CMAKE_SOURCE_DIR}/amx/polymodmul_matmul.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amx/kyber_amx_polymul.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amx/kyber_amx_indcpa.c
)

set(IMPLS neon amx)
set(IMPL_DIRS neon neon)

foreach(PARAMETER_SET IN LISTS KYBER_PARAMETER_SETS)
    foreach(ALLOC stack mmap)
        foreach(IMPL IMPL_DIR IN ZIP_LISTS IMPLS IMPL_DIRS)
            set(LIBRARY ${PARAMETER_SET}_${ALLOC}_${IMPL})

            set(SPEED_PREFIX speed)
            set(SPEED_MATRIXVECTORMUL_PREFIX speed_matrixvectormul)
            set(TEST test_${LIBRARY})

            # Create the static library
            add_library(${LIBRARY} STATIC ${HASH_SOURCES})

            target_include_directories(${LIBRARY} PUBLIC ${PARAMETER_SET}/${IMPL_DIR})
            target_link_libraries(${LIBRARY} PUBLIC neon_rng)

            # Common files and impl-specific files from the NEON dir 
            # (neon uses indcpa.c, amx skips it)
            foreach(SOURCE ${KYBER_SOURCES_COMMON} ${KYBER_SOURCES_${IMPL}})
                target_sources(${LIBRARY} PRIVATE ${PARAMETER_SET}/${IMPL_DIR}/${SOURCE})
            endforeach()

            if(ALLOC STREQUAL stack)
                target_compile_definitions(${LIBRARY} PUBLIC ALLOC_STACK)
            else()
                target_compile_definitions(${LIBRARY} PUBLIC ALLOC_MMAP)
            endif()

            # Include AMX-specific files
            if(IMPL MATCHES amx)
                target_sources(${LIBRARY} PRIVATE ${AMX_SOURCES_amx})
            endif()

            target_include_directories(${LIBRARY} PUBLIC
                ${PARAMETER_SET}/${IMPL_DIR} ${AMX_PATH} ${HASH_PATH} ${RAND_PATH}
                ${CMAKE_CURRENT_SOURCE_DIR}/amx)

            # KEM Speed Benchmark
            set(SPEED ${SPEED_PREFIX}_${LIBRARY})
            add_executable_with_symlink(${SPEED} ${CMAKE_SOURCE_DIR}/speed/speed_kem.c)
            target_compile_options(${SPEED} PRIVATE -DNTESTS=1024 --include=api.h)
            target_include_directories(${SPEED} PRIVATE ${RAND_PATH})
            target_link_libraries(${SPEED} PRIVATE ${LIBRARY} cycles)

            # KEM Speed Benchmark (wall-clock; no kpc/kperf access required)
            set(SPEED_WALLCLOCK ${SPEED_PREFIX}_${LIBRARY}_wallclock)
            add_executable_with_symlink(${SPEED_WALLCLOCK} ${CMAKE_SOURCE_DIR}/speed/speed_kem_wallclock.c)
            target_compile_options(${SPEED_WALLCLOCK} PRIVATE -DNTESTS=1024 --include=api.h)
            target_include_directories(${SPEED_WALLCLOCK} PRIVATE ${RAND_PATH})
            target_link_libraries(${SPEED_WALLCLOCK} PRIVATE ${LIBRARY})

            # Polynomial Multiply Benchmark
            set(SPEED_MATRIXVECTORMUL ${SPEED_MATRIXVECTORMUL_PREFIX}_${LIBRARY})
            set(SPEED_MATRIXVECTORMUL_WALLCLOCK ${SPEED_MATRIXVECTORMUL_PREFIX}_${LIBRARY}_wallclock)
            if(IMPL MATCHES neon)
                add_executable_with_symlink(${SPEED_MATRIXVECTORMUL}
                    ${CMAKE_SOURCE_DIR}/speed/speed_kyber_matrixvectormul_neon.c)
                add_executable_with_symlink(${SPEED_MATRIXVECTORMUL_WALLCLOCK}
                    ${CMAKE_SOURCE_DIR}/speed/speed_kyber_matrixvectormul_neon_wallclock.c)
            else()
                add_executable_with_symlink(${SPEED_MATRIXVECTORMUL}
                    ${CMAKE_SOURCE_DIR}/speed/speed_kyber_matrixvectormul_amx.c)
                add_executable_with_symlink(${SPEED_MATRIXVECTORMUL_WALLCLOCK}
                    ${CMAKE_SOURCE_DIR}/speed/speed_kyber_matrixvectormul_amx_wallclock.c)
            endif()

            target_compile_options(${SPEED_MATRIXVECTORMUL} PRIVATE -DNTESTS=1024 --include=api.h)
            target_include_directories(${SPEED_MATRIXVECTORMUL} PRIVATE ${RAND_PATH})
            target_link_libraries(${SPEED_MATRIXVECTORMUL} PRIVATE ${LIBRARY} cycles)

            target_compile_options(${SPEED_MATRIXVECTORMUL_WALLCLOCK} PRIVATE -DNTESTS=1024 --include=api.h)
            target_include_directories(${SPEED_MATRIXVECTORMUL_WALLCLOCK} PRIVATE ${RAND_PATH})
            target_link_libraries(${SPEED_MATRIXVECTORMUL_WALLCLOCK} PRIVATE ${LIBRARY})

            # correctness test
            add_executable_with_symlink(${TEST} ${CMAKE_SOURCE_DIR}/test/test_kem.cpp)
            target_include_directories(${TEST} PRIVATE ${RAND_PATH})
            target_compile_options(${TEST} PRIVATE -DTEST_NAME=${LIBRARY} --include=api.h)
            target_link_libraries(${TEST} PRIVATE ${LIBRARY} GTest::gtest_main)
            gtest_discover_tests(${TEST} DISCOVERY_TIMEOUT ${GTEST_DISCOVERY_TIMEOUT})
            
        endforeach()
    endforeach()
endforeach()

# Standalone AMX polynomial multiplication test (not parameter-set specific)
add_executable(test_kyber_amx_polymul
    ${CMAKE_SOURCE_DIR}/test/test_kyber_amx_polymul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amx/kyber_amx_polymul.c
    ${CMAKE_SOURCE_DIR}/amx/aux_routines.c
    ${CMAKE_SOURCE_DIR}/amx/polymodmul_matmul.c
)
target_include_directories(test_kyber_amx_polymul PRIVATE
    ${AMX_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/amx
)
target_link_libraries(test_kyber_amx_polymul PRIVATE GTest::gtest_main)
gtest_discover_tests(test_kyber_amx_polymul DISCOVERY_TIMEOUT ${GTEST_DISCOVERY_TIMEOUT})

