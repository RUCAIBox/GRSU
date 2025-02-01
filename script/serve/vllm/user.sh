GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=/path/to/user

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --port "$2" \
  --tensor-parallel-size "${GPU_COUNT}" --swap-space 0 --gpu-memory-utilization 0.95 --max-num-seqs 512 \
  --disable-custom-all-reduce --enforce-eager \
  --trust-remote-code --enable-prefix-caching --seed 42 \
  --disable-log-requests --uvicorn-log-level warning
