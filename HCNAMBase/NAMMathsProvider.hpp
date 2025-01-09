#pragma once

namespace HCNAMBase
{
    // This is taken from https://github.com/jatinchowdhury18/RTNeural-NAM/blob/main/plugin/plugin.h
    struct NAMMathsProvider
    {
#if RTNEURAL_USE_EIGEN
    template <typename Matrix>
    static auto tanh (const Matrix& x)
    {
        // See: math_approx::tanh<3>
        const auto x_poly = x.array() * (1.0f + 0.183428244899f * x.array().square());
        return x_poly.array() * (x_poly.array().square() + 1.0f).array().rsqrt();
    }
//#elif RTNEURAL_USE_XSIMD
//    template <typename T>
//    static T tanh (const T& x)
//    {
//        return math_approx::tanh<3> (x);
//    }
#endif
};
}