cmake_minimum_required(VERSION 3.10)

project(PGDCPU
    VERSION 1.0
    DESCRIPTION "Parallel Gradient Descent CPU"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wfloat-equal)
endif()

find_package(Threads REQUIRED)

set(SRC_FILES
    src/main.cpp
    src/funcs.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/include)

# # --- Optional static analysis via Cppcheck ---
# find_program(CPPCHECK cppcheck)

# if(CPPCHECK)
#         set(CMAKE_CXX_CPPCHECK
#         ${CPPCHECK}
#         --enable=all
#         --inconclusive
#         --std=c++17
#         --force
#         --quiet
#         --inline-suppr
#         --suppress=missingIncludeSystem
#         --suppress=preprocessorErrorDirective
#         --suppress=syntaxError
#         --suppress=unmatchedSuppression
#         --suppress=unmatchedSuppression:*/Eigen/*
#         -I ${PROJECT_SOURCE_DIR}/include
#     )
# else()
#     message(WARNING "Cppcheck not found — static analysis disabled.")
# endif()



add_executable(pgdcpu_multi_cas ${SRC_FILES})
target_compile_options(pgdcpu_multi_cas PRIVATE -O2 -g)
target_link_libraries(pgdcpu_multi_cas PRIVATE Threads::Threads)

# add_executable(pgdcpu_multi_cas_asan ${SRC_FILES})
# target_compile_options(pgdcpu_multi_cas_asan PRIVATE -O1 -g -fsanitize=address -fno-omit-frame-pointer)
# target_link_options(pgdcpu_multi_cas_asan PRIVATE -fsanitize=address)
# target_link_libraries(pgdcpu_multi_cas_asan PRIVATE Threads::Threads)

# add_executable(pgdcpu_multi_cas_tsan ${SRC_FILES})
# target_compile_options(pgdcpu_multi_cas_tsan PRIVATE -O1 -g -fsanitize=thread -fno-omit-frame-pointer)
# target_link_options(pgdcpu_multi_cas_tsan PRIVATE -fsanitize=thread)
# target_link_libraries(pgdcpu_multi_cas_tsan PRIVATE Threads::Threads)

# add_executable(pgdcpu_multi_cas_ubsan ${SRC_FILES})
# target_compile_options(pgdcpu_multi_cas_ubsan PRIVATE -O1 -g -fsanitize=undefined -fno-omit-frame-pointer)
# target_link_options(pgdcpu_multi_cas_ubsan PRIVATE -fsanitize=undefined)
# target_link_libraries(pgdcpu_multi_cas_ubsan PRIVATE Threads::Threads)


# add_executable(pgdcpu_multi_lock ${SRC_FILES})
# target_compile_options(pgdcpu_multi_lock PRIVATE -O2 -g)
# target_link_libraries(pgdcpu_multi_lock PRIVATE Threads::Threads)

# add_executable(pgdcpu_multi_lock_asan ${SRC_FILES})
# target_compile_options(pgdcpu_multi_lock_asan PRIVATE -O1 -g -fsanitize=address -fno-omit-frame-pointer)
# target_link_options(pgdcpu_multi_lock_asan PRIVATE -fsanitize=address)
# target_link_libraries(pgdcpu_multi_lock_asan PRIVATE Threads::Threads)

# add_executable(pgdcpu_multi_lock_tsan ${SRC_FILES})
# target_compile_options(pgdcpu_multi_lock_tsan PRIVATE -O1 -g -fsanitize=thread -fno-omit-frame-pointer)
# target_link_options(pgdcpu_multi_lock_tsan PRIVATE -fsanitize=thread)
# target_link_libraries(pgdcpu_multi_lock_tsan PRIVATE Threads::Threads)

# add_executable(pgdcpu_multi_lock_ubsan ${SRC_FILES})
# target_compile_options(pgdcpu_multi_lock_ubsan PRIVATE -O1 -g -fsanitize=undefined -fno-omit-frame-pointer)
# target_link_options(pgdcpu_multi_lock_ubsan PRIVATE -fsanitize=undefined)
# target_link_libraries(pgdcpu_multi_lock_ubsan PRIVATE Threads::Threads)
