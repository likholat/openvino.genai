// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <string>

#include "text2image/schedulers/types.hpp"
#include "text2image/schedulers/ischeduler.hpp"

namespace ov {
namespace genai {

class EulerDiscreteScheduler : public IScheduler {
public:
    struct Config {
        int32_t num_train_timesteps = 1000;
        float beta_start = 0.0001f, beta_end = 0.02f;
        BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR;
        std::vector<float> trained_betas = {};
        FinalSigmaType final_sigmas_type = FinalSigmaType::ZERO;
        InterpolationType interpolation_type = InterpolationType::LINEAR;
        float sigma_max = 0.0f, sigma_min = 0.0f;
        size_t steps_offset = 0;
        PredictionType prediction_type = PredictionType::EPSILON;
        TimestepSpacing timestep_spacing = TimestepSpacing::LEADING;
        TimestepType timestep_type = TimestepType::DISCRETE;
        bool rescale_betas_zero_snr = false;
        bool use_karras_sigmas = false, use_exponential_sigmas = false, use_beta_sigmas = false;

        Config() = default;
        explicit Config(const std::string& scheduler_config_path);
    };

    explicit EulerDiscreteScheduler(const std::string scheduler_config_path);
    explicit EulerDiscreteScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) override;

private:
    Config m_config;

    std::vector<float> m_alphas_cumprod;
    std::vector<int64_t> m_timesteps;
    std::vector<float> m_sigmas,
};

} // namespace genai
} // namespace ov