// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS FluxTransformer2DModel {
public:
    struct Config {
        size_t in_channels = 64;

        std::vector<size_t> block_out_channels = { 128, 256, 512, 512 };

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit FluxTransformer2DModel(const std::filesystem::path& root_dir);

    FluxTransformer2DModel(const std::filesystem::path& root_dir,
                          const std::string& device,
                          const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    FluxTransformer2DModel(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : FluxTransformer2DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    FluxTransformer2DModel(const FluxTransformer2DModel&);

    const Config& get_config() const;

    FluxTransformer2DModel& reshape(int batch_size, int height, int width, int tokenizer_model_max_length);

    FluxTransformer2DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<FluxTransformer2DModel&, Properties...> compile(const std::string& device,
                                                                                  Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_vae_scale_factor;
};

}  // namespace genai
}  // namespace ov
