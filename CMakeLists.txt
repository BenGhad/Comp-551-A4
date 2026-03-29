cmake_minimum_required(VERSION 3.18)
project(551_A4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(551_A4
        include/dataset.h
        src/dataset.cpp
        src/main.cpp
        include/tokenizer.h
        include/vocab.h
        src/tokenizer.cpp
        src/vocab.cpp
        include/model.h
        src/model.cpp
        include/trainer.h
        src/trainer.cpp
)

target_include_directories(551_A4 PRIVATE include)
target_link_libraries(551_A4 PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET 551_A4 PROPERTY CXX_STANDARD 20)
set_target_properties(551_A4 PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch/lib"
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/libtorch/lib"
)
