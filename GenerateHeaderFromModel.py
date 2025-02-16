import json

def generate_header(data, output_file):
    layers = data['config']['layers']
    weights = data['weights']
    
    with open(output_file, 'w') as f:
        f.write("#pragma once\n\n")
        f.write("#include <array>\n")
        f.write("#include \"NAMMathsProvider.hpp\"\n")
        f.write("#include \"../wavenet/wavenet_model.hpp\"\n")
        f.write("#include \"LayerConfig.hpp\"\n\n")
        f.write("namespace HCNAMBase::Weights {\n\n")
        
        # Write basic config
        f.write(f"constexpr int NUM_LAYERS = {len(layers)};\n")
        #f.write(f"constexpr float HEAD_SCALE = {data['config']['head_scale']};\n\n")
        # HEAD_SCALE is obtained as last weight
             
        f.write("constexpr LayerConfig LAYERS[] = {\n")
        for layer in layers:
            f.write(f"    {{{layer['input_size']}, {layer['condition_size']}, {layer['head_size']}, ")
            f.write(f"{layer['channels']}, {layer['kernel_size']}, {len(layer['dilations'])}, ")
            f.write(f"{str(layer['gated']).lower()}, {str(layer['head_bias']).lower()}}}, \n")
        f.write("};\n\n")

        f.write(f"using NAMModel = typename wavenet::Wavenet_Model<float, 1,\n")
        iLayer = 0
        for layer in layers:
            dilations = ', '.join(map(str, layer['dilations']))
            
            if iLayer != 0:
                f.write(",\n\n")
            
            f.write(f"  wavenet::Layer_Array<float,\n")
            f.write(f"    /*input_size*/     LAYERS[{iLayer}].input_size,\n")
            f.write(f"    /*condition_size*/ LAYERS[{iLayer}].condition_size,\n")
            f.write(f"    /*head_size*/      LAYERS[{iLayer}].head_size,\n")
            f.write(f"    /*channels*/       LAYERS[{iLayer}].channels,\n")
            f.write(f"    /*kernel_size*/    LAYERS[{iLayer}].kernel_size,\n")
            f.write(f"    /*dilations*/      wavenet::Dilations<{dilations}>,\n")
            f.write(f"    /*head_bias*/      LAYERS[{iLayer}].head_bias,\n")
            f.write(f"    /*MathsProvider*/  NAMMathsProvider,\n")
            f.write(f"    /*Activation*/     RTNeural::{str(layer['activation'])}ActivationT<float,\n")
            f.write(f"        /*channels*/       LAYERS[{iLayer}].channels,\n")
            f.write(f"        /*MathsProvider*/  NAMMathsProvider>>")

            iLayer = iLayer + 1

        f.write(">;\n\n")

        # Write weights
        f.write(f"constexpr const std::array<const float, {len(weights)}> WEIGHTS = {{\n")
        for weight in weights:
            f.write(f"    {weight},\n")
        f.write("};\n\n")


        f.write("} // namespace HCNAMBase::Weights\n\n")

with open('model.nam', 'r') as file:
    data = json.load(file)
    generate_header(data, 'HCNAMBase/generated.hpp')
