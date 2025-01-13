#include "weights.hpp"

namespace HCNAMBase
{

    static HCNAMBase::Weights::NAMModel zmoModel;

    void Load() // Temporary
    {
        std::vector<float> weightsVetor(Weights::WEIGHTS.cbegin(), Weights::WEIGHTS.cbegin());

        const auto l1_reschannel = Weights::RECHANNEL<0>; // Hardcoded weights!
        const auto l1_d1_conv = Weights::CONVOLUTION<0,0>();

        zmoModel.load_weights(weightsVetor.begin());

    }

    void Reset(int sample_count)
    {
        zmoModel.prepare(sample_count);
        zmoModel.prewarm();
    }

    void Process(float* in, float* out, int sample_count)
    {
        zmoModel.forward(in, out, sample_count);
    }

} // HCNAMBase
