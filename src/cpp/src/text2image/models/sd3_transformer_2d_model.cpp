// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/sd3_transformer_2d_model.hpp"

SD3Transformer2DModel::Config::Config(const std::string& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "sample_size", sample_size);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "num_layers", num_layers);
    read_json_param(data, "attention_head_dim", attention_head_dim);
    read_json_param(data, "num_attention_heads", num_attention_heads);
    read_json_param(data, "joint_attention_dim", joint_attention_dim);
    read_json_param(data, "caption_projection_dim", caption_projection_dim);
    read_json_param(data, "pooled_projection_dim", pooled_projection_dim);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "pos_embed_max_size", pos_embed_max_size);
}

SD3Transformer2DModel::SD3Transformer2DModel(const std::string root_dir) :
    m_config(root_dir + "/config.json") {
    m_model = ov::Core().read_model(root_dir + "/openvino_model.xml");
}

SD3Transformer2DModel::SD3Transformer2DModel(const std::string& root_dir,
                const std::string& device,
                const ov::AnyMap& properties) :
    SD3Transformer2DModel(root_dir) {
    compile(device, properties);
}

SD3Transformer2DModel::SD3Transformer2DModel(const SD3Transformer2DModel&) = default;

const SD3Transformer2DModel::Config& SD3Transformer2DModel::get_config() const {
    return m_config;
}

// TODO:
SD3Transformer2DModel& SD3Transformer2DModel::reshape(int batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    return *this;
}

SD3Transformer2DModel& SD3Transformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = ov::Core().compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor SD3Transformer2DModel::infer(const ov::Tensor latent,
                                        const ov::Tensor timestep,
                                        const ov::Tensor prompt_embeds,
                                        const ov::Tensor pooled_prompt_embeds) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent);
    m_request.set_tensor("timestep", timestep);
    m_request.set_tensor("encoder_hidden_states", prompt_embeds);
    m_request.set_tensor("pooled_projections", pooled_prompt_embeds);
    m_request.infer();

    return m_request.get_output_tensor();
}
