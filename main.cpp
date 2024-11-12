#include <NAM/dsp.h>
#include <RTNeural/RTNeural.h>
#include <math_approx/math_approx.hpp>
#include <iostream>

#include "wavenet/wavenet_model.hpp"

struct NAMMathsProvider
{
    template <typename Matrix>
    static auto tanh(const Matrix& x)
    {
        // See: math_approx::tanh<3>
        const auto x_poly = x.array() * (1.0f + 0.183428244899f * x.array().square());
        return x_poly.array() * (x_poly.array().square() + 1.0f).array().rsqrt();
    }
};

int main()
{
    const auto model_path { std::string { ROOT_DIR } + "OB1 Mesa DC-5 PM.nam" };

    std::cout << "Loading model from path: " << model_path << std::endl;

    nam::activations::Activation::enable_fast_tanh();

    nam::dspData model_data;
    auto nam_dsp = nam::get_dsp (model_path, model_data);

    using Dilations = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
    wavenet::Wavenet_Model<float,
                           1,
                           wavenet::Layer_Array<float, 1, 1, 8, 16, 3, Dilations, false, NAMMathsProvider>,
                           wavenet::Layer_Array<float, 16, 1, 1, 8, 3, Dilations, true, NAMMathsProvider>>
        rtneural_wavenet;
    rtneural_wavenet.load_weights (model_data.config, model_data.weights);

    nam_dsp->prewarm();
    rtneural_wavenet.prewarm();
    std::cout << std::endl;

    static constexpr size_t N = 2048;
    std::vector<float> input;
    input.resize (N, 0.0f);
    std::vector<float> output_nam;
    output_nam.resize (N, 0.0f);
    std::vector<float> output_rtneural;
    output_rtneural.resize (N, 0.0f);

    for (size_t n = 0; n < input.size(); ++n)
        input[n] = std::sin (3.14f * static_cast<float> (n) * 0.01f);

    auto start = std::chrono::high_resolution_clock::now();
    nam_dsp->process (input.data(), output_nam.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    const auto duration_nam = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();
    std::cout << "NAM: " << duration_nam << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t n = 0; n < input.size(); ++n)
    {
        // nam_dsp->process (input.data() + n, output_nam.data() + n, 1);
        output_rtneural[n] = rtneural_wavenet.forward (input[n]);
        // rtneural_wavenet.reset();
    }
    end = std::chrono::high_resolution_clock::now();
    const auto duration_rtneural = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();
    std::cout << "RTNeural: " << duration_rtneural << std::endl;

    std::cout << "RTNeural is: " << duration_nam / duration_rtneural << "x faster" << std::endl;

    float error_sq_accum = 0.0f;
    for (size_t n = 0; n < input.size(); ++n)
    {
        const auto err = output_nam[n] - output_rtneural[n];
        error_sq_accum += err * err;
    }
    const auto rms_error = std::sqrt (error_sq_accum / static_cast<float> (input.size()));
    std::cout << "RMS error: " << rms_error << std::endl;

    return 0;
}
