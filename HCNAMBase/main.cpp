#include "generated.hpp"

namespace HCNAMBase
{

    static HCNAMBase::NAMModel zmoModel;

    void Load() // Temporary
    {
        std::vector<float> weightsVetor(HCNAMBase::WEIGHTS.cbegin(), HCNAMBase::WEIGHTS.cbegin());

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
