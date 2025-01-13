#pragma once

#include <stddef.h> // size_t

namespace HCNAMBase::Weights
{

struct LayerConfig
{
    const size_t input_size;
    const size_t condition_size;
    const size_t head_size;
    const size_t channels;
    const size_t kernel_size;
    const size_t dilation_count;
    const bool head_bias;
    const bool gated; // not used
   
};
    
} // namespace HCNAMBase::Weights

