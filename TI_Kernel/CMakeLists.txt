cmake_minimum_required(VERSION 3.6)

project(TI_elasticity_kernel)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DIL_OPENBLAS -DIL_BLAS")
set(SOURCE_FILES_DIR ${PROJECT_SOURCE_DIR}/src)

# provide the correct path of OpenBLAS in the following
include_directories(/C/repositories/OpenBLAS/include)
link_directories(/C/repositories/OpenBLAS/lib)

set(SOURCE_FILES
        main.cpp
        src/elasticity_kernel_integration.cpp
        src/mesh.cpp
        src/inputE.cpp
        src/elasticity_kernel_isotropy.cpp
        il/io/numpy/numpy.cpp
        src/AssemblyDDM.cpp
        src/output.h src
        src/Selfcorr.cpp)

add_executable(TI_elasticity_kernel ${SOURCE_FILES})
target_include_directories(TI_elasticity_kernel PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(TI_elasticity_kernel openblas)
