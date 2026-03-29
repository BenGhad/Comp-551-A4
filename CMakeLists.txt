cmake_minimum_required(VERSION 3.18)
project(551_A4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
set_property(TARGET 551_A4 PROPERTY CXX_STANDARD 20)
