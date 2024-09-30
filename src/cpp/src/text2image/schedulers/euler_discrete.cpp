// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <random>
#include <fstream>
#include <iterator>

#include "text2image/schedulers/euler_discrete.hpp"
#include "utils.hpp"
#include "text2image/numpy_utils.hpp"

namespace ov {
namespace genai {

EulerDiscreteScheduler::Config::Config(const std::string& scheduler_config_path) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "num_train_timesteps", num_train_timesteps);
    read_json_param(data, "beta_start", beta_start);
    read_json_param(data, "beta_end", beta_end);
    read_json_param(data, "beta_schedule", beta_schedule);
    read_json_param(data, "trained_betas", trained_betas);
    read_json_param(data, "final_sigmas_type", final_sigmas_type);
    read_json_param(data, "interpolation_type", interpolation_type);
    read_json_param(data, "sigma_max", sigma_max);
    read_json_param(data, "sigma_min", sigma_min);
    read_json_param(data, "steps_offset", steps_offset);
    read_json_param(data, "prediction_type", prediction_type);
    read_json_param(data, "timestep_spacing", timestep_spacing);
    read_json_param(data, "timestep_type", timestep_type);
    read_json_param(data, "rescale_betas_zero_snr", rescale_betas_zero_snr);
    read_json_param(data, "use_karras_sigmas", use_karras_sigmas);
    read_json_param(data, "use_exponential_sigmas", use_exponential_sigmas);
    read_json_param(data, "use_beta_sigmas", use_beta_sigmas);
}

EulerDiscreteScheduler::EulerDiscreteScheduler(const std::string scheduler_config_path) 
    : EulerDiscreteScheduler(Config(scheduler_config_path)) {
}

EulerDiscreteScheduler::EulerDiscreteScheduler(const Config& scheduler_config)
    : m_config(scheduler_config) {

    std::vector<float> alphas, betas;

    using numpy_utils::linspace;

    if (!m_config.trained_betas.empty()) {
        betas = m_config.trained_betas;
    } else if (m_config.beta_schedule == BetaSchedule::LINEAR) {
        betas = linspace<float>(m_config.beta_start, m_config.beta_end, m_config.num_train_timesteps);
    } else if (m_config.beta_schedule == BetaSchedule::SCALED_LINEAR) {
        float start = std::sqrt(m_config.beta_start);
        float end = std::sqrt(m_config.beta_end);
        betas = linspace<float>(start, end, m_config.num_train_timesteps);
        std::for_each(betas.begin(), betas.end(), [] (float & x) { x *= x; });
    } else {
        OPENVINO_THROW("'beta_schedule' must be one of 'LINEAR' or 'SCALED_LINEAR'. Please, add support of other types");
    }

    if (m_config.rescale_betas_zero_snr) {
        using numpy_utils::rescale_zero_terminal_snr;
        rescale_zero_terminal_snr(betas);
    }

    std::transform(betas.begin(), betas.end(), std::back_inserter(alphas), [] (float b) { return 1.0f - b; });

    for (size_t i = 1; i <= alphas.size(); i++) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        m_alphas_cumprod.push_back(alpha_cumprod);
    }

    // check if it works
    if (m_config.rescale_betas_zero_snr) {
        m_alphas_cumprod.back() = std::pow(2, -24);
    }

    // sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
    for (auto it = m_alphas_cumprod.rbegin(); it != m_alphas_cumprod.rend(); ++it) {
        float sigma = std::pow((1 - (*it) / (*it)), 0.5);
        m_sigmas.push_back(sigma);
    }

    auto linspaced = linspace<float>(0.0f, static_cast<float>(m_config.num_train_timesteps - 1), m_config.num_train_timesteps, true);
    for (auto it = linspaced.rbegin(); it != linspaced.rend(); ++it) {
                m_timesteps.push_back(static_cast<int64_t>(std::round(*it)));
    }


    OPENVINO_ASSERT(m_config.timestep_type != TimestepType::CONTINUOUS && m_config.prediction_type != PredictionType::V_PREDICTION,
                    "This case isn't supported: `timestep_type=continuous` and `prediction_type=v_prediction`. Please, add support.");

    // TODO:
    // if (m_config.timestep_type == TimestepType::CONTINUOUS && m_config.prediction_type == PredictionType::V_PREDICTION) {
    //     for (size_t i = 0; i < m_timesteps.size(); ++i) {
    //         m_timesteps[i] = 0.25f * std::log(m_sigmas[i]);
    //     }
    // }

    // torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    m_sigmas.push_back(0);
}

void EulerDiscreteScheduler::set_timesteps(size_t num_inference_steps) {
    m_timesteps.clear();

    OPENVINO_ASSERT(num_inference_steps <= m_config.num_train_timesteps,
                    "`num_inference_steps` cannot be larger than `m_config.num_train_timesteps`");

}

std::map<std::string, ov::Tensor> EulerDiscreteScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    size_t timestep = get_timesteps()[inference_step];

    return {};
}

std::vector<int64_t> EulerDiscreteScheduler::get_timesteps() const {

    return m_timesteps;
}

float EulerDiscreteScheduler::get_init_noise_sigma() const {
    return 1.0f;
}

void EulerDiscreteScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    if(m_step_index == -1)
        m_step_index = 0;

    float sigma = m_sigmas[m_step_index];
    float* sample_data = sample.data<float>();
    for (size_t i = 0; i < sample.get_size(); i++) {
        sample_data[i] /= std::pow((std::pow(sigma, 2) + 1), 0.5);
    }
}

} // namespace genai
} // namespace ov
