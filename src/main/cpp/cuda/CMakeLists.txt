cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME HSE_project)
set(CMAKE_CXX_STANDARD 20)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

file(GLOB SOURCE_FILES CudaTest.cu main.cpp)
file(GLOB HEADER_FILES CudaTest.cuh CudaEvents.cuh)

if (APPLE)
    message(STATUS ">>> No Apple support...")

elseif (MSVC OR MSYS OR MINGW)
    message(STATUS ">>> Windows detected")
    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

elseif (UNIX AND NOT APPLE)
    message(STATUS ">>> No Linux support...")