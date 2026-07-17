cmake_minimum_required(VERSION 3.20...4.3)
project(OmegaNativeEngine VERSION 1.4.1 LANGUAGES CXX)

option(OMEGA_ENABLE_OPENMP "Enable OpenMP parallel batch calculations" ON)
option(OMEGA_BUILD_TESTS "Build native engine tests" ON)
option(OMEGA_ENABLE_PYBIND11 "Build optional pybind11 module when pybind11 is installed" OFF)
option(OMEGA_ENABLE_CPPAD "Use externally installed CppAD in future AD modules" OFF)
option(OMEGA_ENABLE_CERES "Detect Ceres Solver for future solver adapters" OFF)
option(OMEGA_ENABLE_IPOPT "Detect Ipopt for future constrained solver adapters" OFF)
option(OMEGA_ENABLE_NLOPT "Detect NLopt for future global solver adapters" OFF)
option(OMEGA_ENABLE_EIGEN "Detect Eigen for future sparse matrix modules" OFF)
option(OMEGA_ENABLE_CUDA "Enable optional CUDA modules when a CUDA compiler exists" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(omega_native SHARED src/native_engine.cpp)
target_include_directories(omega_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(omega_native PRIVATE OMEGA_NATIVE_BUILD=1)

if(MSVC)
    target_compile_options(omega_native PRIVATE /O2 /permissive- /EHsc)
else()
    target_compile_options(omega_native PRIVATE -O3 -fvisibility=hidden)
endif()

if(OMEGA_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(omega_native PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP was requested but was not found. Building serial native engine.")
    endif()
endif()

if(OMEGA_ENABLE_CPPAD)
    find_path(CPPAD_INCLUDE_DIR cppad/cppad.hpp REQUIRED)
    target_include_directories(omega_native PRIVATE ${CPPAD_INCLUDE_DIR})
    target_compile_definitions(omega_native PRIVATE OMEGA_HAS_CPPAD=1)
endif()
if(OMEGA_ENABLE_CERES)
    find_package(Ceres REQUIRED)
    target_link_libraries(omega_native PRIVATE Ceres::ceres)
    target_compile_definitions(omega_native PRIVATE OMEGA_HAS_CERES=1)
endif()
if(OMEGA_ENABLE_IPOPT)
    find_package(Ipopt REQUIRED)
    target_link_libraries(omega_native PRIVATE Ipopt::ipopt)
    target_compile_definitions(omega_native PRIVATE OMEGA_HAS_IPOPT=1)
endif()
if(OMEGA_ENABLE_NLOPT)
    find_package(NLopt REQUIRED)
    target_link_libraries(omega_native PRIVATE NLopt::nlopt)
    target_compile_definitions(omega_native PRIVATE OMEGA_HAS_NLOPT=1)
endif()
if(OMEGA_ENABLE_EIGEN)
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(omega_native PRIVATE Eigen3::Eigen)
    target_compile_definitions(omega_native PRIVATE OMEGA_HAS_EIGEN=1)
endif()
if(OMEGA_ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(omega_native PRIVATE OMEGA_HAS_CUDA=1)
endif()

set_target_properties(omega_native PROPERTIES
    OUTPUT_NAME "omega_native"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(OMEGA_BUILD_TESTS)
    enable_testing()
    add_executable(test_omega_native tests/test_native_engine.cpp)
    target_include_directories(test_omega_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_omega_native PRIVATE omega_native)
    set_target_properties(test_omega_native PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    add_test(NAME omega_native_engine COMMAND test_omega_native)
endif()

install(TARGETS omega_native
    RUNTIME DESTINATION stock_model/native_libs
    LIBRARY DESTINATION stock_model/native_libs
    ARCHIVE DESTINATION stock_model/native_libs
)
