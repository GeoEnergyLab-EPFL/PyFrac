cmake_minimum_required(VERSION 3.10)
project(src)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(src
        AssemblyDDM.cpp
        elasticity_kernel_integration.cpp
        mesh.cpp
        main.cpp
        json.hpp
        inputE.cpp)
