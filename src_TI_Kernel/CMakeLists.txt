cmake_minimum_required(VERSION 3.10)
project(TI_Kernel)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(TI_Kernel
        AssemblyDDM.cpp
        elasticity_kernel_integration.cpp
        mesh.cpp
        main.cpp
        json.hpp
        inputE.cpp)
