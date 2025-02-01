GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=microsoft/phi-4

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --port "$2" \
  --tensor-parallel-size "${GPU_COUNT}" --gpu-memory-utilization 0.95 --swap-space 0 --max-num-seqs 512 \
  --disable-custom-all-reduce --enforce-eager \
  --trust-remote-code --enable-prefix-caching --seed 42 \
  --uvicorn-log-level warning --disable-log-requests
