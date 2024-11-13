include(CPM.cmake)
CPMAddPackage("gh:juce-framework/JUCE#7.0.10")
CPMAddPackage("gh:Chowdhury-DSP/chowdsp_utils#ef7a6460fae67243ea4d904e8104a29a399ba9c2")

set(JUCE_FORMATS AU VST3 Standalone)

juce_add_plugin(NAM_Test_Plugin
    COMPANY_NAME "ChowDSP"
    PLUGIN_MANUFACTURER_CODE Chow
    PLUGIN_CODE Nam2
    FORMATS ${JUCE_FORMATS}
    PRODUCT_NAME NAM_Test_Plugin
    VERSION 0.0.0

    MICROPHONE_PERMISSION_ENABLED TRUE
)
target_link_libraries(NAM_Test_Plugin
    PRIVATE
        juce::juce_audio_utils
        chowdsp::chowdsp_plugin_base
        chowdsp::chowdsp_plugin_state
        chowdsp::chowdsp_gui
        RTNeural
        math_approx
)
target_compile_definitions(NAM_Test_Plugin
    PUBLIC
        JUCE_VST3_CAN_REPLACE_VST2=0
        ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../"
        NAM_SAMPLE_FLOAT=1
)
target_include_directories(NAM_Test_Plugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/RTNeural/modules/Eigen
)
target_sources(NAM_Test_Plugin
    PRIVATE
        plugin.h
        plugin.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/activations.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/convnet.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/dsp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/get_dsp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/lstm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/util.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../deps/NeuralAmpModelerCore/NAM/wavenet.cpp
)
