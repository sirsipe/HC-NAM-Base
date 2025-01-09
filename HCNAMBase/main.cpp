#include "generated.hpp"

namespace HCNAMBase
{

    static HCNAMBase::NAMModel zmoModel;

    void Load() // Temporary
    {
        //zmoModel.load_weights
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
