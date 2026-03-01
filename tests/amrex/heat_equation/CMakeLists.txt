cmake_minimum_required(VERSION 3.24)
project(HeatEquation CXX)

# Find installed AMReX
find_package(AMReX REQUIRED)

# AMReX source tree — for the HeatEquation test sources
if (NOT DEFINED AMREX_SRC_DIR)
    message(FATAL_ERROR "Set AMREX_SRC_DIR to the AMReX source tree root")
endif()

set(HEAT_SRC_DIR "${AMREX_SRC_DIR}/Tests/HeatEquation")

# Use local main.cpp (float-safe for Metal GPU) but headers from AMReX tree
add_executable(heat_equation
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(heat_equation PRIVATE ${HEAT_SRC_DIR})
target_link_libraries(heat_equation PRIVATE AMReX::amrex_3d)
