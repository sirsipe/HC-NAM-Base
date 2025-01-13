#pragma once
#include "generated.hpp"

namespace HCNAMBase::Weights {

template <typename T, size_t OriginalSize, size_t... Indices>
constexpr std::array<T, sizeof...(Indices)> createSubsetFromIndices(const std::array<T, OriginalSize>& original, std::index_sequence<Indices...>) 
{
    return { original[Indices]... };
}

template <typename T, size_t OriginalSize, size_t Start, size_t End>
constexpr auto createSubset(const std::array<T, OriginalSize>& original) 
{
    constexpr size_t SubsetSize = End - Start;
    return createSubsetFromIndices(original, std::make_index_sequence<SubsetSize>{});
}

constexpr size_t RechannelSize(const size_t layerIndex)
{
    return LAYERS[layerIndex].input_size*LAYERS[layerIndex].channels;
}

constexpr size_t ConvSize(const size_t layerIndex)
{
    return LAYERS[layerIndex].channels * LAYERS[layerIndex].channels * LAYERS[layerIndex].kernel_size;
}

constexpr size_t ConvBiasSize(const size_t layerIndex)
{
    return LAYERS[layerIndex].channels;
}

constexpr size_t InputMixinSize(const size_t layerIndex)
{
    return LAYERS[layerIndex].channels*LAYERS[layerIndex].condition_size;
}

constexpr size_t _1x1Size(const size_t layerIndex)
{
    return LAYERS[layerIndex].channels*LAYERS[layerIndex].channels;
}

constexpr size_t _1x1BiasSize(const size_t layerIndex)
{
    return LAYERS[layerIndex].channels;
}

constexpr size_t DilationsSize(const size_t layerIndex)
{
    return LAYERS[layerIndex].dilation_count * 
    (
        ConvSize(layerIndex)
        + ConvBiasSize(layerIndex)
        + InputMixinSize(layerIndex)
        + _1x1Size(layerIndex)
        + _1x1BiasSize(layerIndex)
    );
}


constexpr size_t WeightCount(const size_t layerIndex)
{
  return RechannelSize(layerIndex) + DilationsSize(layerIndex);
}


constexpr size_t FirstWeight(const size_t layer)
{
  if (layer == 0) return 0;
  else return FirstWeight(layer - 1) + WeightCount(layer -1);
}

template<size_t layer> constexpr std::array<const float, RechannelSize(layer)> RECHANNEL = 
        createSubset<const float, WEIGHTS.size(), FirstWeight(layer), RechannelSize(layer)>(WEIGHTS); 

template<size_t layer, size_t dilationLayer> constexpr std::array<const float, ConvSize(layer)> CONVOLUTION() 
{
    constexpr size_t startIndex = 
        RechannelSize(layer) 
        + DilationsSize(layer) * dilationLayer;
    return createSubset<const float, WEIGHTS.size(), startIndex, startIndex + ConvSize(layer)>(WEIGHTS);  // TODO: kernel weighs are in reverse order!!
} 

template<size_t layer, size_t dilationLayer> constexpr std::array<const float, ConvBiasSize(layer)> CONVOLUTION_BIAS()
{
    constexpr size_t startIndex = 
        RechannelSize(layer) 
        + DilationsSize(layer) * dilationLayer
        + ConvSize(layer);
    return createSubset<const float, WEIGHTS.size(), startIndex, ConvBiasSize(layer)>(WEIGHTS); 
}

template<size_t layer, size_t dilationLayer> constexpr std::array<const float, InputMixinSize(layer)> INPUT_MIXIN()
{
    constexpr size_t startIndex = 
        RechannelSize(layer) 
        + DilationsSize(layer) * dilationLayer
        + ConvSize(layer)
        + ConvBiasSize(layer);

    return createSubset<const float, WEIGHTS.size(), startIndex, startIndex + InputMixinSize(layer)>(WEIGHTS); 
} 

template<size_t layer, size_t dilationLayer> constexpr std::array<const float, _1x1Size(layer)> _1X1()
{
    constexpr size_t startIndex = 
        RechannelSize(layer) 
        + DilationsSize(layer) * dilationLayer
        + ConvSize(layer)
        + ConvBiasSize(layer)
        + InputMixinSize(layer);

    return createSubset<const float, WEIGHTS.size(), startIndex, startIndex + _1x1Size(layer)>(WEIGHTS); 
}

template<size_t layer, size_t dilationLayer> constexpr std::array<const float, _1x1BiasSize(layer)> _1X1BIAS()
{
    constexpr size_t startIndex = 
        RechannelSize(layer) 
        + DilationsSize(layer) * dilationLayer
        + ConvSize(layer)
        + ConvBiasSize(layer)
        + InputMixinSize(layer)
        + _1x1Size(layer);

    return createSubset<const float, WEIGHTS.size(), startIndex, startIndex + _1x1BiasSize(layer)>(WEIGHTS); 
} 

constexpr float HEAD_SCALE()
{
    return WEIGHTS[WEIGHTS.size() -1]; 
} 

}; // namespace HCNAMBase
