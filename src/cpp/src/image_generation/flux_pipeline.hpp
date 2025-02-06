// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "utils.hpp"

namespace {

ov::Tensor pack_latents(const ov::Tensor latents, size_t batch_size, size_t num_channels_latents, size_t height, size_t width) {
    size_t h_half = height / 2, w_half = width / 2;

    // Reshape to (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    ov::Shape final_shape = {batch_size, h_half * w_half, num_channels_latents * 4};
    ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), final_shape);

    OPENVINO_ASSERT(latents.get_size() == permuted_latents.get_size(), "Incorrect target shape, tensors must have the same sizes");

    float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    // Permute to (0, 2, 4, 1, 3, 5)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_channels_latents; ++c) {
            for (size_t h2 = 0; h2 < h_half; ++h2) {
                for (size_t w2 = 0; w2 < w_half; ++w2) {
                    size_t base_src_index = (b * num_channels_latents + c) * height * width + (h2 * 2 * width + w2 * 2);
                    size_t base_dst_index = (b * h_half * w_half + h2 * w_half + w2) * num_channels_latents * 4 + c * 4;

                    dst_data[base_dst_index] = src_data[base_src_index];
                    dst_data[base_dst_index + 1] = src_data[base_src_index + 1];
                    dst_data[base_dst_index + 2] = src_data[base_src_index + width];
                    dst_data[base_dst_index + 3] = src_data[base_src_index + width + 1];
                }
            }
        }
    }

    return permuted_latents;
}

ov::Tensor unpack_latents(const ov::Tensor& latents, size_t height, size_t width, size_t vae_scale_factor) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape[0], channels = latents_shape[2];

    height /= vae_scale_factor;
    width /= vae_scale_factor;

    // latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    size_t h_half = height / 2;
    size_t w_half = width / 2;
    size_t c_quarter = channels / 4;

    // Reshape to (batch_size, channels // (2 * 2), height, width)
    ov::Shape final_shape = {batch_size, c_quarter, height, width};
    ov::Tensor permuted_latents(latents.get_element_type(), final_shape);

    OPENVINO_ASSERT(latents.get_size() == permuted_latents.get_size(), "Incorrect target shape, tensors must have the same sizes");

    const float* src_data = latents.data<float>();
    float* dst_data = permuted_latents.data<float>();

    // Permutation to (0, 3, 1, 4, 2, 5)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c4 = 0; c4 < c_quarter; ++c4) {
            for (size_t h2 = 0; h2 < h_half; ++h2) {
                for (size_t w2 = 0; w2 < w_half; ++w2) {
                    size_t base_reshaped_index = (((b * h_half + h2) * w_half + w2) * c_quarter + c4) * 4;
                    size_t base_final_index = (b * c_quarter * height * width) + (c4 * height * width) + (h2 * 2 * width + w2 * 2);

                    dst_data[base_final_index] = src_data[base_reshaped_index];
                    dst_data[base_final_index + 1] = src_data[base_reshaped_index + 1];
                    dst_data[base_final_index + width] = src_data[base_reshaped_index + 2];
                    dst_data[base_final_index + width + 1] = src_data[base_reshaped_index + 3];
                }
            }
        }
    }

    return permuted_latents;
}

ov::Tensor prepare_latent_image_ids(size_t batch_size, size_t height, size_t width) {
    ov::Tensor latent_image_ids(ov::element::f32, {height * width, 3});
    auto* data = latent_image_ids.data<float>();

    std::fill(data, data + height * width * 3, 0.0);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            data[(i * width + j) * 3 + 1] = static_cast<float>(i);
            data[(i * width + j) * 3 + 2] = static_cast<float>(j);
        }
    }

    return latent_image_ids;
}

}  // namespace

namespace ov {
namespace genai {

class FluxPipeline : public DiffusionPipeline {
public:
    explicit FluxPipeline(PipelineType pipeline_type) : DiffusionPipeline(pipeline_type) {
        // TODO: support GPU as well
        const std::string device = "CPU";

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
            const bool do_normalize = true, do_binarize = false, gray_scale_source = false;
            m_image_processor = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, gray_scale_source);
            m_image_resizer = std::make_shared<ImageResizer>(device, ov::element::u8, "NHWC", ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
        }

        if (m_pipeline_type == PipelineType::INPAINTING) {
            bool do_normalize = false, do_binarize = true;
            m_mask_processor_rgb = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, false);
            m_mask_processor_gray = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, true);
            m_mask_resizer = std::make_shared<ImageResizer>(device, ov::element::f32, "NCHW", ov::op::v11::Interpolate::InterpolateMode::NEAREST);
        }
    }

    FluxPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) : FluxPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string t5_text_encoder = data["text_encoder_2"][1].get<std::string>();
        if (t5_text_encoder == "T5EncoderModel") {
            m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_2");
        } else {
            OPENVINO_THROW("Unsupported '", t5_text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(PipelineType pipeline_type,
                 const std::filesystem::path& root_dir,
                 const std::string& device,
                 const ov::AnyMap& properties)
        : FluxPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string t5_text_encoder = data["text_encoder_2"][1].get<std::string>();
        if (t5_text_encoder == "T5EncoderModel") {
            m_t5_text_encoder = std::make_shared<T5EncoderModel>(root_dir / "text_encoder_2", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", t5_text_encoder, "' text encoder type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "' Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(PipelineType pipeline_type,
                 const CLIPTextModel& clip_text_model,
                 const T5EncoderModel& t5_text_model,
                 const FluxTransformer2DModel& transformer,
                 const AutoencoderKL& vae)
        : FluxPipeline(pipeline_type) {
        m_clip_text_encoder = std::make_shared<CLIPTextModel>(clip_text_model);
        m_t5_text_encoder = std::make_shared<T5EncoderModel>(t5_text_model);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        m_transformer = std::make_shared<FluxTransformer2DModel>(transformer);
        initialize_generation_config("FluxPipeline");
    }

    FluxPipeline(PipelineType pipeline_type, const FluxPipeline& pipe) :
        FluxPipeline(pipe) {
        OPENVINO_ASSERT(!pipe.is_inpainting_model(), "Cannot create ",
            pipeline_type == PipelineType::TEXT_2_IMAGE ? "'Text2ImagePipeline'" : "'Image2ImagePipeline'", " from InpaintingPipeline with inpainting model");

        m_pipeline_type = pipeline_type;
        initialize_generation_config("FluxPipeline");
    }

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        m_clip_text_encoder->reshape(1);
        m_t5_text_encoder->reshape(1, m_generation_config.max_sequence_length);
        m_transformer->reshape(num_images_per_prompt, height, width, m_generation_config.max_sequence_length);

        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        m_clip_text_encoder->compile(device, properties);
        m_t5_text_encoder->compile(device, properties);
        m_vae->compile(device, properties);
        m_transformer->compile(device, properties);
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        // encode_prompt
        std::string prompt_2_str = generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;

        m_clip_text_encoder->infer(positive_prompt, {}, false);
        ov::Tensor pooled_prompt_embeds = m_clip_text_encoder->get_output_tensor(1);
        ov::Tensor prompt_embeds = m_t5_text_encoder->infer(prompt_2_str, "", false, generation_config.max_sequence_length);

        pooled_prompt_embeds = numpy_utils::repeat(pooled_prompt_embeds, generation_config.num_images_per_prompt);
        prompt_embeds = numpy_utils::repeat(prompt_embeds, generation_config.num_images_per_prompt);

        // text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        ov::Shape text_ids_shape = {prompt_embeds.get_shape()[1], 3};
        ov::Tensor text_ids(ov::element::f32, text_ids_shape);
        std::fill_n(text_ids.data<float>(), text_ids.get_size(), 0.0f);

        const size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        ov::Tensor latent_image_ids = prepare_latent_image_ids(generation_config.num_images_per_prompt, height / 2, width / 2);

        if (m_transformer->get_config().guidance_embeds) {
            ov::Tensor guidance = ov::Tensor(ov::element::f32, {generation_config.num_images_per_prompt});
            std::fill_n(guidance.data<float>(), guidance.get_size(), static_cast<float>(generation_config.guidance_scale));
            m_transformer->set_hidden_states("guidance", guidance);
        }

        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds);
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("txt_ids", text_ids);
        m_transformer->set_hidden_states("img_ids", latent_image_ids);
    }

    std::vector<float> get_timesteps(size_t num_inference_steps, float strength) {
        float init_timestep = std::min(static_cast<float>(num_inference_steps) * strength, static_cast<float>(num_inference_steps));
        size_t t_start = static_cast<size_t>(std::max(static_cast<float>(num_inference_steps) - init_timestep, 0.0f));

        std::vector<float> timesteps, m_scheduler_timesteps = m_scheduler->get_float_timesteps();
        for (size_t i = t_start; i < m_scheduler_timesteps.size(); ++i) {
            timesteps.push_back(m_scheduler_timesteps[i]);
        }

        m_scheduler->set_begin_index(t_start);

        return timesteps;
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt,
                               num_channels_latents,
                               height,
                               width};
        ov::Tensor latent, noise, proccesed_image, image_latents;

        if (initial_image) {
            proccesed_image = m_image_resizer->execute(initial_image, generation_config.height, generation_config.width);
            proccesed_image = m_image_processor->execute(proccesed_image);

            image_latents = m_vae->encode(proccesed_image, generation_config.generator);
            noise = generation_config.generator->randn_tensor(latent_shape);

            latent = ov::Tensor(image_latents.get_element_type(), image_latents.get_shape());
            image_latents.copy_to(latent);

            m_scheduler->scale_noise(latent, m_latent_timestep, noise);
            latent = pack_latents(latent, generation_config.num_images_per_prompt, num_channels_latents, height, width);

            if (m_pipeline_type == PipelineType::INPAINTING) {
                noise = pack_latents(noise, generation_config.num_images_per_prompt, num_channels_latents, height, width);
                image_latents = pack_latents(image_latents, generation_config.num_images_per_prompt, num_channels_latents, height, width);
            }
        } else {
            noise = generation_config.generator->randn_tensor(latent_shape);
            latent = pack_latents(noise, generation_config.num_images_per_prompt, num_channels_latents, height, width);
        }

        return std::make_tuple(latent, proccesed_image, image_latents, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        OPENVINO_THROW("LORA adapters are not implemented for FLUX pipeline yet");
    }

    std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor mask_image, ov::Tensor processed_image, const ImageGenerationConfig& generation_config) {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'prepare_mask_latents' can be called for inpainting pipeline only");

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Shape target_shape = processed_image.get_shape();

        // Prepare mask latent variables
        ov::Tensor mask_condition = m_image_resizer->execute(mask_image, generation_config.height, generation_config.width);
        std::shared_ptr<IImageProcessor> mask_processor = mask_condition.get_shape()[3] == 1 ? m_mask_processor_gray : m_mask_processor_rgb;
        mask_condition = mask_processor->execute(mask_condition);

        size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        size_t height = generation_config.height / vae_scale_factor;
        size_t width = generation_config.width / vae_scale_factor;

        // resize mask to shape of latent space
        ov::Tensor mask = m_mask_resizer->execute(mask_condition, height, width);
        mask = numpy_utils::repeat(mask, generation_config.num_images_per_prompt);

        // Create masked image:
        // masked_image = init_image * (mask_condition < 0.5)
        ov::Tensor masked_image(ov::element::f32, processed_image.get_shape());
        const float * mask_condition_data = mask_condition.data<const float>();
        const float * processed_image_data = processed_image.data<const float>();
        float * masked_image_data = masked_image.data<float>();
        for (size_t i = 0, plane_size = mask_condition.get_shape()[2] * mask_condition.get_shape()[3]; i < mask_condition.get_size(); ++i) {
            masked_image_data[i + 0 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 0 * plane_size] : 0.0f;
            masked_image_data[i + 1 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 1 * plane_size] : 0.0f;
            masked_image_data[i + 2 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 2 * plane_size] : 0.0f;
        }

        ov::Tensor masked_image_latent;
        // TODO: support is_inpainting_model() == true
        // masked_image_latent = m_vae->encode(masked_image, generation_config.generator);
        // // masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        // float * masked_image_latent_data = masked_image_latent.data<float>();
        // for (size_t i = 0; i < masked_image_latent.get_size(); ++i) {
        //     masked_image_latent_data[i] = (masked_image_latent_data[i] - m_vae->get_config().shift_factor) * m_vae->get_config().scaling_factor;
        // }
        // masked_image_latent = pack_latents(masked_image_latent, generation_config.num_images_per_prompt, num_channels_latents, height, width);

        // mask.repeat(1, num_channels_latents, 1, 1)
        auto repeat_mask = [](const ov::Tensor& mask, size_t num_channels_latents) -> ov::Tensor {
            const ov::Shape& mask_shape = mask.get_shape();
            OPENVINO_ASSERT(mask_shape.size() == 4 && mask_shape[1] == 1, "Mask must have shape (batch_size, 1, height, width)");

            size_t batch_size = mask_shape[0], height = mask_shape[2], width = mask_shape[3];
            size_t spatial_size = height * width;

            ov::Shape target_shape = {batch_size, num_channels_latents, height, width};
            ov::Tensor repeated_mask(mask.get_element_type(), target_shape);

            const float* src_data = mask.data<float>();
            float* dst_data = repeated_mask.data<float>();

            for (size_t b = 0; b < batch_size; ++b) {
                const float* src_batch = src_data + b * spatial_size;  // Pointer to batch start
                float* dst_batch = dst_data + b * num_channels_latents * spatial_size;

                for (size_t c = 0; c < num_channels_latents; ++c) {
                    std::memcpy(dst_batch + c * spatial_size, src_batch, spatial_size * sizeof(float));
                }
            }

            return repeated_mask;
        };

        ov::Tensor repeated_mask = repeat_mask(mask, num_channels_latents);
        ov::Tensor mask_packed = pack_latents(repeated_mask, generation_config.num_images_per_prompt, num_channels_latents, height, width);

        return std::make_tuple(mask_packed, masked_image_latent);
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        // Use callback if defined
        std::function<bool(size_t, size_t, ov::Tensor&)> callback = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback = callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>();
        }

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();

        if (m_custom_generation_config.height < 0)
            compute_dim(m_custom_generation_config.height, initial_image, 1 /* assume NHWC */);
        if (m_custom_generation_config.width < 0)
            compute_dim(m_custom_generation_config.width, initial_image, 2 /* assume NHWC */);

        check_inputs(m_custom_generation_config, initial_image);

        compute_hidden_states(positive_prompt, m_custom_generation_config);

        size_t image_seq_len = (m_custom_generation_config.height / vae_scale_factor / 2) *
                               (m_custom_generation_config.width / vae_scale_factor / 2);
        float mu = m_scheduler->calculate_shift(image_seq_len);
        float linspace_end = 1.0f / m_custom_generation_config.num_inference_steps;
        std::vector<float> sigmas = numpy_utils::linspace<float>(1.0f, linspace_end, m_custom_generation_config.num_inference_steps, true);
        m_scheduler->set_timesteps_with_sigma(sigmas, mu);

        std::vector<float> timesteps;
        if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
            timesteps = m_scheduler->get_float_timesteps();
        } else {
            timesteps = get_timesteps(m_custom_generation_config.num_inference_steps, m_custom_generation_config.strength);
        }
        m_latent_timestep = timesteps[0];

        ov::Tensor latents, processed_image, image_latent, noise;
        std::tie(latents, processed_image, image_latent, noise) = prepare_latents(initial_image, m_custom_generation_config);

        // prepare mask latents
        ov::Tensor mask, masked_image_latent;
        if (m_pipeline_type == PipelineType::INPAINTING) {
            std::tie(mask, masked_image_latent) = prepare_mask_latents(mask_image, processed_image, m_custom_generation_config);
        }

        // 6. Denoising loop
        ov::Tensor timestep(ov::element::f32, {1});
        float* timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            timestep_data[0] = timesteps[inference_step] / 1000.0f;

            ov::Tensor noise_pred_tensor = m_transformer->infer(latents, timestep);

            auto scheduler_step_result = m_scheduler->step(noise_pred_tensor, latents, inference_step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            if (m_pipeline_type == PipelineType::INPAINTING) {
                blend_latents(latents, image_latent, mask, noise, inference_step);
            }

            if (callback && callback(inference_step, timesteps.size(), latents)) {
                return ov::Tensor(ov::element::u8, {});
            }
        }

        latents = unpack_latents(latents, m_custom_generation_config.height, m_custom_generation_config.width, vae_scale_factor);
        return m_vae->decode(latents);
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        ov::Tensor unpacked_latent = unpack_latents(latent,
                                     m_custom_generation_config.height,
                                     m_custom_generation_config.width,
                                     m_vae->get_vae_scale_factor());
        return m_vae->decode(unpacked_latent);
    }

private:
    bool is_inpainting_model() const {
        assert(m_transformer != nullptr);
        assert(m_vae != nullptr);
        return m_transformer->get_config().in_channels == (m_vae->get_config().latent_channels * 2 + 1);
    }

    void compute_dim(int64_t & generation_config_value, ov::Tensor initial_image, int dim_idx) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();

        if (generation_config_value < 0)
            generation_config_value = transformer_config.m_default_sample_size * vae_scale_factor;
    }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_transformer != nullptr);
        assert(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        m_generation_config.height = transformer_config.m_default_sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.m_default_sample_size * vae_scale_factor;

        if (class_name == "FluxPipeline" || class_name == "FluxImg2ImgPipeline" || class_name == "FluxInpaintPipeline" ) {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
                m_generation_config.guidance_scale = 3.5f;
                m_generation_config.num_inference_steps = 28;
                m_generation_config.strength = 1.0f;
            } else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_generation_config.guidance_scale = 7.0f;
                m_generation_config.num_inference_steps = 28;
                m_generation_config.strength = 0.6f;
            }
            m_generation_config.max_sequence_length = 512;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_transformer != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ",
                        vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "T5's 'max_sequence_length' must be less or equal to 512");

        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by FluxPipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by FluxPipeline");

        if ((m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) && initial_image) {
            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
        }
    }

    void blend_latents(ov::Tensor latents,
                       const ov::Tensor image_latent,
                       const ov::Tensor mask,
                       const ov::Tensor noise,
                       size_t inference_step) {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'blend_latents' can be called for inpainting pipeline only");
        OPENVINO_ASSERT(image_latent.get_shape() == latents.get_shape(),
                        "Shapes for current", latents.get_shape(), "and initial image latents ", image_latent.get_shape(), " must match");

        ov::Tensor init_latents_proper(image_latent.get_element_type(), image_latent.get_shape());
        image_latent.copy_to(init_latents_proper);

        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        if (inference_step < timesteps.size() - 1) {
            float noise_timestep = timesteps[inference_step + 1];
            m_scheduler->scale_noise(init_latents_proper, noise_timestep, noise);
        }

        float * latents_data = latents.data<float>();
        const float * mask_data = mask.data<float>();
        const float * init_latents_proper_data = init_latents_proper.data<float>();
        for (size_t i = 0; i < latents.get_size(); ++i) {
            latents_data[i] = (1.0f - mask_data[i]) * init_latents_proper_data[i] + mask_data[i] * latents_data[i];
        }
    }

    std::shared_ptr<FluxTransformer2DModel> m_transformer = nullptr;
    std::shared_ptr<CLIPTextModel> m_clip_text_encoder = nullptr;
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder = nullptr;
    std::shared_ptr<AutoencoderKL> m_vae = nullptr;
    std::shared_ptr<IImageProcessor> m_image_processor = nullptr, m_mask_processor_rgb = nullptr, m_mask_processor_gray = nullptr;
    std::shared_ptr<ImageResizer> m_image_resizer = nullptr, m_mask_resizer = nullptr;
    ImageGenerationConfig m_custom_generation_config;

    float m_latent_timestep = -1;
};

}  // namespace genai
}  // namespace ov
