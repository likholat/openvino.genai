#  Who What Benchmark (WWB) - Simple Accuracy Benchmarking Tool for Generative AI models
The main idea of the benchmark is to estimate the similarity score between embedding computed by for data generated by two models, for example, baseline and optimized. In general, this can be the data created with the model inferred with different tools. Thus, this similarity allows to understand how different data in general.

WWB provides default datasets for the supported use cases. However, it is relatively easy to plug and use custom datasets.


## Features

* Command-line interface for Hugging Face and OpenVINO models and API to support broader inference backends.
* Simple and quick accuracy test for compressed, quantized, pruned, distilled LLMs. It works with any model that supports HuggingFace Transformers text generation API including:
    * HuggingFace Transformers compressed models via [Bitsandbytes](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)
    * [OpenVINO](https://github.com/openvinotoolkit/openvino) and [NNCF](https://github.com/openvinotoolkit/nncf) via [Optimum-Intel](https://github.com/huggingface/optimum-intel) and OpenVINO [GenAI](https://github.com/openvinotoolkit/openvino.genai)
    * [GPTQ](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig) via HuggingFace API
    * Llama.cpp via [BigDL-LLM](https://github.com/intel-analytics/BigDL/tree/main/python/llm)
    * Support of custom datasets of the user choice
* Validation of text-to-image pipelines. Computes similarity score between generated images with Diffusers library, Optimum-Intel, and OpenVINO GenAI via `Text2ImageEvaluator` class.
* Validation of Visual Language pipelines. Computes similarity score between generated images with Diffusers library, Optimum-Intel, and OpenVINO GenAI via `VisualTextEvaluator` class.

### Installation
To install WWB and its dependencies, follow these steps:
1. Set up a Python virtual environment (recommended):
```
    python -m venv eval_env
    source eval_env/bin/activate
```
2. Install WWB from the source directory:
```
    pip install .
```
To install WWB with nightly builds of openvino, openvino-tokenizers, and openvino-genai, use the following command:
```
PIP_PRE=1 \
PIP_EXTRA_INDEX_URL=https://storage.openvinotoolkit.org/simple/wheels/nightly \
pip install .
```

## Usage
### Compare Text-generation Models (LLMs)
```sh
# Collect ground truth from the baseline Hugging Face Transformer model 
wwb --base-model microsoft/Phi-3-mini-4k-instruct --gt-data gt.csv --model-type text --hf

# Convert model to Optimum-Intel (quantized to 8-bit by default)
optimum-cli export openvino -m microsoft/Phi-3-mini-4k-instruct phi-3-openvino

# Measure similarity metric for Optimum-OpenVINO inference backend
wwb --target-model phi-3-openvino --gt-data gt.csv --model-type text

# Measure similarity metric for OpenVINO GenAI inference backend
wwb --target-model phi-3-openvino --gt-data gt.csv --model-type text --genai
```

> **NOTE**: use --verbose option for debug to see the outputs with the largest difference.

### Compare Text-to-image models
```sh
# Export model with 8-bit quantized weights to OpenVINO
optimum-cli export openvino -m SimianLuo/LCM_Dreamshaper_v7 --weight-format int8 sd-lcm-int8
# Collect the references and save the mappling in the .csv file. 
# Reference images will be stored in the "reference" subfolder under the same path with .csv.
wwb --base-model SimianLuo/LCM_Dreamshaper_v7--gt-data lcm_test/gt.csv --model-type text-to-image --hf
# Compute the metric
# Target images will be stored in the "target" subfolder under the same path with .csv.
wwb --target-model sd-lcm-int8 --gt-data lcm_test/gt.csv --model-type text-to-image --genai
```

### Compare Visual Language Models (VLMs)
```sh
# Export FP16 model to OpenVINO
optimum-cli export openvino -m llava-hf/llava-v1.6-mistral-7b-hf  --weight-format int8 llava-int8
# Collect the references and save the mappling in the .csv file. 
# Reference images will be stored in the "reference" subfolder under the same path with .csv.
wwb --base-model llava-hf/llava-v1.6-mistral-7b-hf --gt-data llava_test/gt.csv --model-type visual-text --hf
# Compute the metric
# Target images will be stored in the "target" subfolder under the same path with .csv.
wwb --target-model llava-int8 --gt-data llava_test/gt.csv --model-type visual-text --genai
```

### Compare Visual Language Models with LoRA (VLMs)
```sh
# Export FP16 model to OpenVINO
optimum-cli export openvino -m black-forest-labs/FLUX.1-dev FLUX.1-dev-fp

# Collect the references and save the mappling in the .csv file.
# Reference images will be stored in the "reference" subfolder under the same path with .csv.
wwb --base-model black-forest-labs/FLUX.1-dev --gt-data flux.1-dev/gt.csv --model-type text-to-image --adapters Octree/flux-schnell-lora Shakker-Labs/FLUX.1-dev-LoRA-add-details --alphas 0.1 0.9 --hf
# Compute the metric
# Target images will be stored in the "target" subfolder under the same path with .csv.
wwb --target-model FLUX.1-dev-fp --gt-data flux.1-dev/gt.csv --model-type text-to-image --adapters flux-schnell-lora.safetensors FLUX-dev-lora-add_details.safetensors --alphas 0.1 0.9 --genai
```

### API
The API provides a way to access to investigate the worst generated text examples.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import whowhatbench

model_id = "facebook/opt-1.3b"
base_small = AutoModelForCausalLM.from_pretrained(model_id)
optimized_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

evaluator = whowhatbench.TextEvaluator(base_model=base_small, tokenizer=tokenizer)
metrics_per_prompt, metrics = evaluator.score(optimized_model)

metric_of_interest = "similarity"
print(metric_of_interest, ": ", metrics["similarity"][0])

worst_examples = evaluator.worst_examples(top_k=5, metric=metric_of_interest)
print("Metric: ", metric_of_interest)
for e in worst_examples:
    print("\t=========================")
    print("\tPrompt: ", e["prompt"])
    print("\tBaseline Model:\n ", "\t" + e["source_model"])
    print("\tOptimized Model:\n ", "\t" + e["optimized_model"])

```

Use your own list of prompts to compare (e.g. from a dataset):
```python
from datasets import load_dataset
val = load_dataset("lambada", split="validation[20:40]")
prompts = val["text"]
...
metrics_per_prompt, metrics = evaluator.score(optimized_model, test_data=prompts)
```

### Advaned CLI usage

```sh
wwb --help

# Run ground truth generation for uncompressed model on the first 32 samples from squad dataset
# Ground truth will be saved in llama_2_7b_squad_gt.csv file
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_squad_gt.csv --dataset squad --split validation[:32] --dataset-field question

# Run comparison with compressed model on the first 32 samples from squad dataset
wwb --target-model /home/user/models/Llama_2_7b_chat_hf_int8 --gt-data llama_2_7b_squad_gt.csv --dataset squad --split validation[:32] --dataset-field question

# Output will be like this
#   similarity        FDT        SDT  FDT norm  SDT norm
# 0    0.972823  67.296296  20.592593  0.735127  0.151505

# Run ground truth generation for uncompressed model on internal set of questions
# Ground truth will be saved in llama_2_7b_squad_gt.csv file
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv

# Run comparison with compressed model on internal set of questions
wwb --target-model /home/user/models/Llama_2_7b_chat_hf_int8 --gt-data llama_2_7b_wwb_gt.csv

# Use --num-samples to control the number of samples
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv --num-samples 10

# Use -v for verbose mode to see the difference in the results
wwb --target-model /home/user/models/Llama_2_7b_chat_hf_int8 --gt-data llama_2_7b_wwb_gt.csv  --num-samples 10 -v

# Use --hf AutoModelForCausalLM to instantiate the model from model_id/folder
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv --hf

# Use --language parameter to control the language of prompts
# Autodetection works for basic Chinese models 
wwb --base-model meta-llama/Llama-2-7b-chat-hf --gt-data llama_2_7b_wwb_gt.csv --hf
```

### Supported metrics

* `similarity` - averaged similarity measured by neural network trained for sentence embeddings. The best is 1.0, the minimum is 0.0, higher-better.
* `FDT` - Average position of the first divergent token between sentences generated by different LLMs. The worst is 0, higher-better. [Paper.](https://arxiv.org/abs/2311.01544)
* `FDT norm` - Average share of matched tokens until first divergent one between sentences generated by different LLMs. The best is 1, higher-better.[Paper.](https://arxiv.org/abs/2311.01544)
* `SDT` - Average number of divergent tokens in the evaluated outputs between sentences generated by different LLMs. The best is 0, lower-better. [Paper.](https://arxiv.org/abs/2311.01544)
* `SDT norm` - Average share of divergent tokens in the evaluated outputs between sentences generated by different LLMs. The best is 0, the maximum is 1, lower-better. [Paper.](https://arxiv.org/abs/2311.01544)

### Notes

* The generation of ground truth on uncompressed model must be run before comparison with compressed model.
* WWB uses [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for similarity measurement but you can use other similar network.
