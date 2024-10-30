// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <ctime>

#include "text2image/diffusion_pipeline.hpp"
#include "text2image/numpy_utils.hpp"
#include "utils.hpp"

namespace {

// ov::Tensor pack_latents(ov::Tensor latents,
//                         size_t batch_size,
//                         size_t num_channels_latents,
//                         size_t height,
//                         size_t width) {
//     ov::Shape shape1 = {batch_size, num_channels_latents, height / 2, 2, width / 2, 2};
//     latents.set_shape(shape1);

//     // permute to (0, 2, 4, 1, 3, 5)
//     ov::Shape permuted_shape = {batch_size, height / 2, width / 2, num_channels_latents, 2, 2};
//     ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), permuted_shape);

//     auto* src_data = latents.data<float>();  // adjust type if necessary
//     auto* dst_data = permuted_latents.data<float>();

//     for (int b = 0; b < batch_size; ++b) {
//         for (int h2 = 0; h2 < height / 2; ++h2) {
//             for (int w2 = 0; w2 < width / 2; ++w2) {
//                 for (int c = 0; c < num_channels_latents; ++c) {
//                     for (int h3 = 0; h3 < 2; ++h3) {
//                         for (int w3 = 0; w3 < 2; ++w3) {
//                             int src_index = ((b * num_channels_latents + c) * (height / 2) + h2) * 2 * (width / 2) * 2 +
//                                             (h3 * width / 2 + w2) * 2 + w3;
//                             int dst_index = ((b * (height / 2) + h2) * (width / 2) + w2) * num_channels_latents * 4 +
//                                             (c * 4 + h3 * 2 + w3);
//                             dst_data[dst_index] = src_data[src_index];
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     ov::Shape shape2 = {batch_size, (height / 2) * (width / 2), num_channels_latents * 4};
//     permuted_latents.set_shape(shape2);

//     return permuted_latents;
// }

ov::Tensor pack_latents(const ov::Tensor& latents, size_t batch_size, size_t num_channels_latents, size_t height, size_t width) {
    // Check if the input tensor has the correct shape
    // if (latents.get_shape().size() != 4 || latents.get_shape()[0] != batch_size || latents.get_shape()[1] != num_channels_latents || latents.get_shape()[2] != height || latents.get_shape()[3] != width) {
    //     throw std::invalid_argument("Latents tensor shape does not match expected dimensions.");
    // }

    // Calculate the new dimensions after packing
    size_t new_height = height / 2;
    size_t new_width = width / 2;

    // Reshape the tensor to [batch_size, num_channels_latents, new_height, 2, new_width, 2]
    ov::Shape reshaped_shape{batch_size, num_channels_latents, new_height, 2, new_width, 2};
    ov::Tensor reshaped_latents(ov::element::f32, reshaped_shape, latents.data<float>()); // Assuming dtype is float32

    // Prepare the output shape
    ov::Shape output_shape{batch_size, new_height * new_width, num_channels_latents * 4};

    // Create the output tensor
    ov::Tensor packed_latents(ov::element::f32, output_shape);

    // Perform the packing operation
    auto* input_data = reshaped_latents.data<float>();
    auto* output_data = packed_latents.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < new_height; ++h) {
            for (size_t w = 0; w < new_width; ++w) {
                for (size_t c = 0; c < num_channels_latents; ++c) {
                    // Map the input tensor to the output tensor
                    size_t input_index = b * num_channels_latents * height * width + c * height * width + h * 2 * width + w * 2;
                    size_t output_index = b * (new_height * new_width) * (num_channels_latents * 4) + h * new_width + w;

                    // Populate the packed tensor
                    output_data[output_index * (num_channels_latents * 4) + c * 4 + 0] = input_data[input_index];
                    output_data[output_index * (num_channels_latents * 4) + c * 4 + 1] = input_data[input_index + 1];
                    output_data[output_index * (num_channels_latents * 4) + c * 4 + 2] = input_data[input_index + width];
                    output_data[output_index * (num_channels_latents * 4) + c * 4 + 3] = input_data[input_index + width + 1];
                }
            }
        }
    }

    return packed_latents;
}

ov::Tensor unpack_latents(ov::Tensor latents, size_t height, size_t width, size_t vae_scale_factor) {
    ov::Shape latents_shape = latents.get_shape();
    size_t batch_size = latents_shape[0];
    size_t num_patches = latents_shape[1];
    size_t channels = latents_shape[2];

    height = height / vae_scale_factor;
    width = width / vae_scale_factor;

    // latents to (batch_size, height // 2, width // 2, channels // 4, 2, 2)
    ov::Shape shape1 = {batch_size, height / 2, width / 2, channels / 4, 2, 2};
    latents.set_shape(shape1);

    // Permute to (0, 3, 1, 4, 2, 5)
    ov::Shape permuted_shape = {batch_size, channels / 4, height / 2, 2, width / 2, 2};
    ov::Tensor permuted_latents = ov::Tensor(latents.get_element_type(), permuted_shape);

    auto* src_data = latents.data<float>();
    auto* dst_data = permuted_latents.data<float>();

    // Manually perform the permutation:
    for (int b = 0; b < batch_size; ++b) {
        for (int c4 = 0; c4 < channels / 4; ++c4) {
            for (int h2 = 0; h2 < height / 2; ++h2) {
                for (int w2 = 0; w2 < width / 2; ++w2) {
                    for (int h3 = 0; h3 < 2; ++h3) {
                        for (int w3 = 0; w3 < 2; ++w3) {
                            int src_index = ((b * (height / 2) + h2) * (width / 2) + w2) * (channels / 4) * 4 +
                                            (c4 * 4 + h3 * 2 + w3);
                            int dst_index =
                                ((b * (channels / 4) + c4) * (height / 2) * 2 + h2 * 2 + h3) * (width / 2) * 2 +
                                w2 * 2 + w3;
                            dst_data[dst_index] = src_data[src_index];
                        }
                    }
                }
            }
        }
    }

    // (batch_size, channels // (2 * 2), height, width)
    ov::Shape final_shape = {batch_size, channels / 4, height, width};
    permuted_latents.set_shape(final_shape);

    return permuted_latents;
}

ov::Tensor prepare_latent_image_ids(size_t batch_size, size_t height, size_t width) {
    size_t latent_height = height / 2;
    size_t latent_width = width / 2;
    size_t channels = 3;

    ov::Shape shape{latent_height, latent_width, channels};
    ov::Tensor latent_image_ids(ov::element::f32, shape);
    float* data = latent_image_ids.data<float>();
    std::fill_n(data, latent_height * latent_width * channels, 0.0f);

    for (size_t i = 0; i < latent_height; ++i) {
        for (size_t j = 0; j < latent_width; ++j) {
            data[(i * latent_width + j) * channels + 1] = static_cast<float>(i);
            data[(i * latent_width + j) * channels + 2] = static_cast<float>(j);
        }
    }

    // Reshape the tensor to [latent_height * latent_width, channels]
    ov::Shape reshaped_shape{latent_height * latent_width, channels};
    latent_image_ids = ov::Tensor(ov::element::f32, reshaped_shape, data);

    return latent_image_ids;
}


}  // namespace

namespace ov {
namespace genai {

class Text2ImagePipeline::FluxPipeline : public Text2ImagePipeline::DiffusionPipeline {
public:
    explicit FluxPipeline(const std::filesystem::path& root_dir) {
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
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer");
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) {
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
            m_vae_decoder = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        const std::string transformer = data["transformer"][1].get<std::string>();
        if (transformer == "FluxTransformer2DModel") {
            m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", transformer, "'Transformer type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    FluxPipeline(const CLIPTextModel& clip_text_model,
                 const T5EncoderModel& t5_text_model,
                 const FluxTransformer2DModel& transformer,
                 const AutoencoderKL& vae_decoder)
        : m_clip_text_encoder(std::make_shared<CLIPTextModel>(clip_text_model)),
          m_t5_text_encoder(std::make_shared<T5EncoderModel>(t5_text_model)),
          m_vae_decoder(std::make_shared<AutoencoderKL>(vae_decoder)),
          m_transformer(std::make_shared<FluxTransformer2DModel>(transformer)) {}

    // TODO
    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        // check_image_size(height, width);

        // const size_t batch_size_multiplier =
        //     do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Transformer accepts 2x batch in case of CFG
        // m_clip_text_encoder_1->reshape(batch_size_multiplier);
        // m_clip_text_encoder_2->reshape(batch_size_multiplier);
        // m_transformer->reshape(num_images_per_prompt * batch_size_multiplier,
        //                        height,
        //                        width,
        //                        m_clip_text_encoder_1->get_config().max_position_embeddings);
        // m_vae_decoder->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        m_clip_text_encoder->compile(device, properties);
        m_t5_text_encoder->compile(device, properties);
        m_vae_decoder->compile(device, properties);
        m_transformer->compile(device, properties);
    }

    ov::Tensor generate(const std::string& positive_prompt, const ov::AnyMap& properties) override {
        using namespace numpy_utils;
        GenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();

        std::cout << "vae_scale_factor " << vae_scale_factor << std::endl;

        if (generation_config.height < 0)
            generation_config.height = m_default_sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = m_default_sample_size * vae_scale_factor;

        check_inputs(generation_config);

        // encode_prompt
        std::string prompt_2_str =
            generation_config.prompt_2 != std::nullopt ? *generation_config.prompt_2 : positive_prompt;

        m_clip_text_encoder->infer(positive_prompt, "", false);
        size_t idx_pooler_output = 1;
        ov::Tensor pooled_prompt_embeds_out = m_clip_text_encoder->get_output_tensor(idx_pooler_output);

        ov::Tensor prompt_embeds_out = m_t5_text_encoder->infer(positive_prompt);

        ov::Tensor pooled_prompt_embeds, prompt_embeds;
        if (generation_config.num_images_per_prompt == 1) {
            pooled_prompt_embeds = pooled_prompt_embeds_out;
            prompt_embeds = prompt_embeds_out;
        } else {
            pooled_prompt_embeds =
                tensor_batch_copy(pooled_prompt_embeds_out, generation_config.num_images_per_prompt, 1);
            prompt_embeds = tensor_batch_copy(prompt_embeds_out, generation_config.num_images_per_prompt, 1);
        }

        // text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        ov::Shape text_ids_shape = {prompt_embeds.get_shape()[1], 3};
        ov::Tensor text_ids(ov::element::f32, text_ids_shape);
        std::fill_n(text_ids.data<float>(), text_ids_shape[0] * text_ids_shape[1], 0.0f);

        // std::cout << "pooled_prompt_embeds" << std::endl;
        // for (int i = 0; i<10; ++i){
        //     std::cout << pooled_prompt_embeds.data<float>()[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "prompt_embeds" << std::endl;
        // for (int i = 0; i<10; ++i){
        //     std::cout << prompt_embeds.data<float>()[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "text_ids" << std::endl;
        // for (int i = 0; i<10; ++i){
        //     std::cout << text_ids.data<float>()[i] << " ";
        // }
        // std::cout << std::endl;

        size_t num_channels_latents = m_transformer->get_config().in_channels / 4;
        // ov::Shape latent_inp_shape{generation_config.num_images_per_prompt,
        //                            num_channels_latents,
        //                            generation_config.height / vae_scale_factor,
        //                            generation_config.width / vae_scale_factor};

        size_t height = 2 * generation_config.height / vae_scale_factor;
        size_t width = 2 * generation_config.width / vae_scale_factor;
        ov::Shape latent_inp_shape{generation_config.num_images_per_prompt, height/2 * width/2, num_channels_latents * 4};

        ov::Tensor latents_inp(ov::element::f32, latent_inp_shape);
        std::generate_n(latents_inp.data<float>(), latents_inp.get_size(), [&]() -> float {
            return generation_config.random_generator->next() * m_scheduler->get_init_noise_sigma();
        });

        ov::Tensor latents = latents_inp;

        

        // ov::Tensor latents = pack_latents(latents_inp,
        //                                   generation_config.num_images_per_prompt,
        //                                   num_channels_latents,
        //                                   height,
        //                                   width);

        
        ov::Tensor latent_image_ids = prepare_latent_image_ids(generation_config.num_images_per_prompt, height, width);

        std::cout << "latents" << std::endl;
        for (int i = 0; i<10; ++i){
            std::cout << latents.data<float>()[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "latent_image_ids" << std::endl;
        for (int i = 0; i<10; ++i){
            std::cout << latent_image_ids.data<float>()[i] << " ";
        }
        std::cout << std::endl;

        // std::cout << "pooled_projections " << pooled_prompt_embeds.get_shape() << std::endl;
        // std::cout << "encoder_hidden_states " << prompt_embeds.get_shape() << std::endl;
        // std::cout << "txt_ids" << text_ids.get_shape() << std::endl;
        // std::cout << "img_ids" << latent_image_ids.get_shape() << std::endl;

        m_transformer->set_hidden_states("pooled_projections", pooled_prompt_embeds);
        m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
        m_transformer->set_hidden_states("txt_ids", text_ids);
        m_transformer->set_hidden_states("img_ids", latent_image_ids);

        // TODO: mu = calculate_shift(...)
        float mu = 0.63f;

        float linspace_end = 1.0f / generation_config.num_inference_steps;
        std::vector<float> sigmas = linspace<float>(1.0f, linspace_end, generation_config.num_inference_steps, true);

        // std::cout << "sigmas" << std::endl;
        // for (auto i : sigmas){
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        m_scheduler->set_timesteps_with_sigma(sigmas, mu);
        std::vector<float> timesteps = m_scheduler->get_float_timesteps();
        size_t num_inference_steps = timesteps.size();

        // std::cout << "timesteps" << std::endl;
        // for (auto i : timesteps){
        //     std::cout << i << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "num_inference_steps " << num_inference_steps << std::endl;

        // 6. Denoising loop
        ov::Tensor noisy_residual_tensor(ov::element::f32, {});
        size_t timestep_size = latents.get_shape()[0];
        ov::Tensor timestep(ov::element::f32, {timestep_size});

        for (size_t inference_step = 0; inference_step < num_inference_steps; ++inference_step) {
            float* timestep_data = timestep.data<float>();
            std::fill_n(timestep_data, timestep_size, (timesteps[inference_step] / 1000));

            // std::cout << "latents" << latents.get_shape() << std::endl;
            // std::cout << "timestep" << timesteps[inference_step] / 1000 << std::endl;

            ov::Tensor noise_pred_tensor = m_transformer->infer(latents, timestep);

            std::cout << "noise_pred_tensor" << std::endl;
            for (int i = 0; i<10; ++i){
                std::cout << noise_pred_tensor.data<float>()[i] << " ";
            }
            std::cout << std::endl;

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latents, inference_step);
            latents = scheduler_step_result["latent"];

            std::cout << "latents" << std::endl;
            for (int i = 0; i<10; ++i){
                std::cout << latents.data<float>()[i] << " ";
            }
            std::cout << std::endl;
        }

        latents = unpack_latents(latents, generation_config.height, generation_config.width, vae_scale_factor);
        std::cout << "latents 1" << std::endl;
        for (int i = 0; i<10; ++i){
            std::cout << latents.data<float>()[i] << " ";
        }
        std::cout << std::endl;
        float* latent_data = latents.data<float>();
        for (size_t i = 0; i < latents.get_size(); ++i) {
            latent_data[i] = (latent_data[i] / m_vae_decoder->get_config().scaling_factor) +
                             m_vae_decoder->get_config().shift_factor;
        }

        std::cout << "latents 2" << std::endl;
        for (int i = 0; i<10; ++i){
            std::cout << latents.data<float>()[i] << " ";
        }
        std::cout << std::endl;

        return m_vae_decoder->infer(latents);
    }

private:
    // bool do_classifier_free_guidance(float guidance_scale) const {
    //     return guidance_scale >= 1.0;
    // }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_transformer != nullptr);
        assert(m_vae_decoder != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();

        m_default_sample_size = 128;
        m_generation_config.height = m_default_sample_size * vae_scale_factor;
        m_generation_config.width = m_default_sample_size * vae_scale_factor;

        if (class_name == "FluxPipeline") {
            m_generation_config.guidance_scale = 3.5f;
            m_generation_config.num_inference_steps = 28;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_transformer != nullptr);
        const size_t vae_scale_factor = m_transformer->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) && (width % vae_scale_factor == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by",
                        vae_scale_factor);
    }

    void check_inputs(const GenerationConfig& generation_config) const override {
        check_image_size(generation_config.width, generation_config.height);

        const char* const pipeline_name = "Flux";

        OPENVINO_ASSERT(generation_config.negative_prompt.empty(), "Negative prompt is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt,
                        "Negative prompt 2 is not used by ",
                        pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt,
                        "Negative prompt 3 is not used by ",
                        pipeline_name);

        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
    }

    std::shared_ptr<FluxTransformer2DModel> m_transformer;
    std::shared_ptr<CLIPTextModel> m_clip_text_encoder;
    // TODO:
    std::shared_ptr<T5EncoderModel> m_t5_text_encoder;
    std::shared_ptr<AutoencoderKL> m_vae_decoder;

private:
    size_t m_default_sample_size;
};

}  // namespace genai
}  // namespace ov
