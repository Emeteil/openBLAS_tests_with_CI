cmake_minimum_required(VERSION 3.16)
project(openBLAS_tests C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NOFORTRAN ON CACHE BOOL "Build without Fortran support" FORCE)
set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "Build without LAPACK (only BLAS/CBLAS)" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared library" FORCE)
set(DYNAMIC_ARCH OFF CACHE BOOL "Include support for multiple CPU targets" FORCE)

add_subdirectory(OpenBLAS)

add_executable(main main.cpp)
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS
    ${CMAKE_CURRENT_BINARY_DIR}/OpenBLAS
    ${CMAKE_CURRENT_BINARY_DIR}/OpenBLAS/generated
)
target_link_libraries(main PRIVATE openblas)
