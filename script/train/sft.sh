GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

master_port=$2

model_name=meta-llama/Llama-3.1-8B-Instruct
model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
response_template='<|start_header_id|>assistant<|end_header_id|>'

batch_size=4
learning_rate=1e-6
min_lr_rate=0.1
warmup_steps=1000
weight_decay=0.01

dataset_names=(
  "redial+inspired+reward_no-cot+item_n-1+neg_n-1+n_neg_per_pos-1+feedback_item_n-10+neg_n-1+Llama-3.3-70B-Instruct+1219"
)
dataset_paths=(
  "dataset/inspired/reward_no-cot_data/item_n-1+neg_n-1+n_neg_per_pos-1.jsonl dataset/inspired/feedback_data/item_n-10+neg_n-1+Llama-3.3-70B-Instruct+1219.jsonl dataset/redial/reward_no-cot_data/item_n-1+neg_n-1+n_neg_per_pos-1.jsonl dataset/redial/feedback_data/item_n-10+neg_n-1+Llama-3.3-70B-Instruct+1219.jsonl"
)
dataset_size=-1
max_seq_length=3072

length=${#dataset_names[@]}
for (( i=0; i<$length; i++ )); do
  dataset_name=${dataset_names[$i]}
  dataset_path=${dataset_paths[$i]}

  run_dir_suffix=${dataset_name}+n-${dataset_size}/${model_name}
  log_dir=log/train/${run_dir_suffix}
  mkdir -p ${log_dir}

  run_name=bs-${batch_size}+lr-${learning_rate}+wd-${weight_decay}+warmup-${warmup_steps}+min_lr_rate-${min_lr_rate}+len-${max_seq_length}

  output_dir=ckpt/${run_dir_suffix}/${run_name}
  echo ${output_dir}

  CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node="${GPU_COUNT}" --master-port="${master_port}" \
    -m src.train.sft \
    --seed 42 \
    --model_name_or_path ${model_name_or_path} \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --response_template ${response_template} \
    --dataset_path ${dataset_path} \
    --dataset_size ${dataset_size} \
    --dataset_num_proc 32 \
    --max_seq_length ${max_seq_length} \
    --save_strategy no \
    --save_only_model \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --max_grad_norm 1 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\"min_lr_rate\": ${min_lr_rate}}" \
    --warmup_steps ${warmup_steps} \
    --gradient_checkpointing True \
    --deepspeed script/train/ds_z3_bf16.json \
    --logging_steps 1 \
    --report_to tensorboard \
  &> ${log_dir}/${run_name}.log
done