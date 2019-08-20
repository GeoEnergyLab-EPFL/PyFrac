cmake_minimum_required(VERSION 3.10)
project(TI_kernel)

set(CMAKE_CXX_STANDARD 11)
set(GCC_CXX_FLAGS ${GCC_CXX_FLAGS} "-static-libgcc -static-libstdc++ -static")

include_directories(.)

add_executable(TI_kernel
        AssemblyDDM.cpp
        elasticity_kernel_integration.cpp
        mesh.cpp
        main.cpp
        json.hpp
        inputE.cpp)
