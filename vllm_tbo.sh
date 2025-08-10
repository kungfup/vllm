CUDA_VISIBLE_DEVICES=0,1 VLLM_ALL2ALL_BACKEND=pplx vllm serve /home/yzh/model/Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --enable-microbatching \
  --enforce-eager \
  --max-num-batched-tokens 2048