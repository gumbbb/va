#!/usr/bin/env bash
set -ex

MODEL="TheBloke/Llama-2-7B-Chat-GPTQ"

LLAMA_EXAMPLES="/opt/TensorRT-LLM/examples/llama"
TRT_LLM_MODELS="/data/models/tensorrt_llm"

: "${FORCE_BUILD:=off}"

MODEL_DIR=$(huggingface-downloader $MODEL)
output_dir="$TRT_LLM_MODELS/Llama-2-7B-Chat-GPTQ"
engine_dir="$output_dir/engines"

if [ ! -f $output_dir/*.safetensors ] || [ "$FORCE_BUILD" = "on" ]; then
    mkdir -p "$output_dir"
    python3 $LLAMA_EXAMPLES/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$output_dir" \
        --dtype float16 \
        --quant_ckpt_path "$MODEL_DIR/model.safetensors" \
        --use_weight_only \
        --weight_only_precision int4_gptq \
        --group_size 128 \
        --per_group
fi

if [ ! -f $engine_dir/*.engine ] || [ "$FORCE_BUILD" = "on" ]; then
    trtllm-build \
        --checkpoint_dir "$output_dir" \
        --output_dir "$engine_dir" \
        --gemm_plugin auto \
        --log_level verbose \
        --max_batch_size 1 \
        --max_num_tokens 512 \
        --max_seq_len 512 \
        --max_input_len 128
fi

python3 $LLAMA_EXAMPLES/../run.py \
    --max_input_len=128 \
    --max_output_len=128 \
    --max_attention_window_size 256 \
    --max_tokens_in_paged_kv_cache=256 \
    --tokenizer_dir "$MODEL_DIR" \
    --engine_dir "$engine_dir"

python3 /opt/TensorRT-LLM/benchmarks/python/benchmark.py \
    -m dec \
    --engine_dir "$engine_dir" \
    --quantization int4_weight_only_gptq \
    --batch_size 1 \
    --input_output_len "16,128;32,128;64,128;128,128" \
    --log_level verbose \
    --enable_cuda_graph \
    --warm_up 2 \
    --num_runs 3 \
    --duration 10

echo "Done. To start OpenAI server, run:"
echo "jetson-containers run dustynv/tensorrt_llm:0.12-r36.4.0 \\"
echo "  python3 /opt/TensorRT-LLM/examples/apps/openai_server.py $engine_dir"
