
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Geo-Energy Laboratory, 2016-2022.  All rights reserved.
# See the LICENSE.TXT file for more details.
#
# Contributors:
# Brice Lecampion
# Haseeb Zia
# Carlo Peruzzo

cmake_minimum_required(VERSION 3.6)

project(TI_elasticity_kernel)

set(CMAKE_CXX_STANDARD 11)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)


################################################################################
# OpenBLAS
################################################################################

# provide the correct path of folders containg OpenBLAS include files and libraries in the following. Note that if you
# are using msys2 on windows, you have to set it for UNIX.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIL_OPENBLAS -DIL_BLAS")
set(CMAKE_OPENBLAS_LIBRARIES "openblas")

if (UNIX AND NOT APPLE)
    # For windows, provide the correct path of folders containing OpenBLAS include files and libraries in the following.
    # Note that we are using msys2 on windows, so we are setting it for UNIX.
    #include_directories(/C/repositories/OpenBLAS/include)
    #link_directories(/C/repositories/OpenBLAS/build/lib)
elseif(APPLE)  # for APPLE
    include_directories(/usr/local/opt/openblas/include)
    link_directories(/usr/local/opt/openblas/lib)
endif()


################################################################################
# parallel framework
################################################################################
set(IL_OPENMP = 1)

# For OpenMP
if(IL_OPENMP)

    if(APPLE)
        MESSAGE("-- System is APPLE")
        if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            MESSAGE("-- Compiler is Clang")
            set(CMAKE_C_COMPILER /usr/local/opt/llvm/bin/clang)
            set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/opt/llvm/include -fopenmp=libomp -DNUMBEROFTHREADS=4")
            MESSAGE("-- OPENMP ENABLED")
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            MESSAGE("-- Compiler is GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            MESSAGE("-- OPENMP ENABLED")
        endif()
    elseif(UNIX)
        MESSAGE("-- System is UNIX")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            MESSAGE("-- Compiler is Clang")
            # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DNUMBEROFTHREADS=4")
            # link_directories(/opt/clang-3.9.0/lib)
            MESSAGE("-- OPENMP ENABLED")
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            MESSAGE("-- Compiler is GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp -DNUMBEROFTHREADS=4")
            MESSAGE("-- OPENMP ENABLED")
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            MESSAGE("-- Compiler is Intel")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
            MESSAGE("-- OPENMP ENABLED")
        endif()
    elseif(WIN32)
        MESSAGE("-- System is WIN32")
        if(MINGW)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DNUMBEROFTHREADS=4")
            MESSAGE("-- OPENMP ENABLED")
        endif()
    endif()
endif()
################################################################################

set(SOURCE_FILES_DIR ${PROJECT_SOURCE_DIR}/src)

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

if(APPLE)
        if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            target_link_libraries(TI_elasticity_kernel  ${CMAKE_OPENBLAS_LIBRARIES} -lpthread -lm -ldl -L/usr/local/opt/llvm/lib)
        endif()
else()
            target_link_libraries(TI_elasticity_kernel  ${CMAKE_OPENBLAS_LIBRARIES})
endif()

target_compile_definitions(TI_elasticity_kernel PRIVATE IL_OPENBLAS=1 IL_BLAS=1)
target_include_directories(TI_elasticity_kernel PRIVATE ${CMAKE_SOURCE_DIR})


################################################################################
# notes for mac OS users (parallel framework)
################################################################################
#- OPEN MP
#This code uses openMP to parallelize some parts.
#If we are using mac OS, now (2019), Clang and Clang++ (*1) doesn't have openMP.
#You need to install llvm compiler with openMP. We use Homebrew to get the packages.
#Open a terminal and type:

# brew update
# brew install llvm libomp

# You will obtain the following message at the end:
# ___________________________________________________________________
# To use the bundled libc++ please add the following LDFLAGS:
#   LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"

# llvm is keg-only, which means it was not symlinked into /usr/local,
# because macOS already provides this software and installing another version in
# parallel can cause all kinds of trouble.

# If you need to have llvm first in your PATH run:
#   echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.bash_profile

# For compilers to find llvm you may need to set:
#   export LDFLAGS="-L/usr/local/opt/llvm/lib"
#   export CPPFLAGS="-I/usr/local/opt/llvm/include"
# ___________________________________________________________________

#So, you need to add those lines at the end of your .bash_profile (*2)

# LDFLAGS="-L/usr/local/opt/llvm/lib"
# CPPFLAGS="-I/usr/local/opt/llvm/include"
# PATH="/usr/local/opt/llvm/bin:$PATH"

# After your .bash_profile has been modified you need to source it.
# Type in your terminal:

# source .bash_profile


#(*1) Usually Clang and Clang++ are located in /usr/bin/
#(*2) To locate your .bash_profile open a terminal and type:
#      cd
#      ls -la

