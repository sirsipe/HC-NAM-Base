#pragma once

#include <RTNeural/RTNeural.h>

template <typename ModelType>
void load_tcn_model (ModelType& model, const nlohmann::json& model_json) {

    int input_size = model_json["model_data"]["input_size"];

    auto jsonTo1DVector = [](const nlohmann::json& j) -> std::vector<float> {
        std::vector<float> vec = j.get<std::vector<float>>();
        return vec;
    };

    auto jsonTo2DVector = [jsonTo1DVector](const nlohmann::json& j) -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> vec;
        for (const auto& sub_j : j) {
            vec.push_back(jsonTo1DVector(sub_j));
        }
        return vec;
    };

    auto reverse_kernels = [](std::vector<std::vector<std::vector<float>>>& conv_weights)
    {
        for (auto& channel_weights : conv_weights)
        {
            for (auto& kernel : channel_weights)
            {
                std::reverse(kernel.begin(), kernel.end());
            }
        }
    };

    auto jsonTo3DVector = [jsonTo2DVector, reverse_kernels](const nlohmann::json& j) -> std::vector<std::vector<std::vector<float>>> {
        std::vector<std::vector<std::vector<float>>> vec;
        for (const auto& sub_j : j) {
            vec.push_back(jsonTo2DVector(sub_j));
            // std::cout << "sub_j: " << sub_j << std::endl;
        }
        reverse_kernels (vec);
        return vec;
    };

    // Set weights and biases for the first and last convolution layers
    model.first_conv.setWeights(jsonTo3DVector(model_json["/state_dict/first_conv.weight"_json_pointer]));
    model.first_conv.setBias(jsonTo1DVector(model_json["/state_dict/first_conv.bias"_json_pointer]));

    model.last_conv.setWeights(jsonTo3DVector(model_json["/state_dict/last_conv.weight"_json_pointer]));
    model.last_conv.setBias(jsonTo1DVector(model_json["/state_dict/last_conv.bias"_json_pointer]));

    // Set weights and biases for each block and layer using forEachInTuple
    for (size_t block_idx = 0; block_idx < model.blocks.size(); ++block_idx) {
        auto& block = model.blocks[block_idx];
        std::string block_prefix = "/state_dict/blocks." + std::to_string(block_idx) + ".layers.";

        RTNeural::modelt_detail::forEachInTuple(
            [&](auto& layer, size_t layer_idx) {
                std::string layer_prefix = block_prefix + std::to_string(layer_idx) + ".";

                layer.conditioning.setWeights(jsonTo2DVector(model_json[nlohmann::json::json_pointer(layer_prefix + "conditioning.weight")]));
                layer.conditioning.setBias(jsonTo1DVector(model_json[nlohmann::json::json_pointer(layer_prefix + "conditioning.bias")]).data());
                layer.conv.setWeights(jsonTo3DVector(model_json[nlohmann::json::json_pointer(layer_prefix + "conv.weight")]));
                layer.conv.setBias(jsonTo1DVector(model_json[nlohmann::json::json_pointer(layer_prefix + "conv.bias")]));
                layer.resi_con.setWeights(jsonTo3DVector(model_json[nlohmann::json::json_pointer(layer_prefix + "resi_con.weight")]));
                layer.resi_con.setBias(jsonTo1DVector(model_json[nlohmann::json::json_pointer(layer_prefix + "resi_con.bias")]));
            },
            block.layers
        );
    }
}