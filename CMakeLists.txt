cmake_minimum_required(VERSION 3.10)
project(HCNAMBase)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(deps/RTNeural)

add_library(HCNAMBase SHARED)
target_sources(HCNAMBase
    PRIVATE
        HCNAMBase/main.cpp
)
target_compile_definitions(HCNAMBase
    PRIVATE
        RTNEURAL_USE_EIGEN=1
)

target_include_directories(HCNAMBase
    PRIVATE
        deps/RTNeural/modules/Eigen
)

target_link_libraries(HCNAMBase
    PRIVATE
        RTNeural
)
