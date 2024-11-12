cmake_minimum_required(VERSION 3.29)
project(RTNeural_NAM)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(deps/RTNeural)
add_subdirectory(deps/NeuralAmpModelerCore)

add_executable(RTNeural_NAM)
target_sources(RTNeural_NAM
    PRIVATE
        main.cpp
        deps/NeuralAmpModelerCore/NAM/activations.cpp
        deps/NeuralAmpModelerCore/NAM/convnet.cpp
        deps/NeuralAmpModelerCore/NAM/dsp.cpp
        deps/NeuralAmpModelerCore/NAM/get_dsp.cpp
        deps/NeuralAmpModelerCore/NAM/lstm.cpp
        deps/NeuralAmpModelerCore/NAM/util.cpp
        deps/NeuralAmpModelerCore/NAM/wavenet.cpp
)
target_compile_definitions(RTNeural_NAM
    PRIVATE
        ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
        NAM_SAMPLE_FLOAT=1
)

target_include_directories(RTNeural_NAM
    PRIVATE
        deps/NeuralAmpModelerCore
)
target_link_libraries(RTNeural_NAM
    PRIVATE
        RTNeural
)
