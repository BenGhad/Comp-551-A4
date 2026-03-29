cmake_minimum_required(VERSION 3.18)
project(551_A4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)

add_executable(551_A4
        src/dataset.cpp
        src/main.cpp
        src/tokenizer.cpp
        src/vocab.cpp
        src/model.cpp
        src/trainer.cpp
)

target_include_directories(551_A4 PRIVATE include)
target_link_libraries(551_A4 PRIVATE "${TORCH_LIBRARIES}")
target_compile_definitions(551_A4 PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_precompile_headers(551_A4 PRIVATE <torch/torch.h>)
set_property(TARGET 551_A4 PROPERTY CXX_STANDARD 20)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
