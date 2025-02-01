from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, enable_full_determinism, set_seed, AutoModelForCausalLM
from trl import (
    ModelConfig,
    SFTConfig,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map, SFTTrainer,
)
from trl.commands.cli_utils import TrlParser

from src.train.sft_utils import DataCollatorForCompletionOnlyLM, get_response_template_ids


@dataclass
class SFTScriptArguments(SFTConfig):
    response_template: str = None
    dataset_path: List[str] = field(default_factory=list)
    dataset_size: int = -1
    gradient_checkpointing_use_reentrant: bool = False
    test: bool = None


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_and_config()
    args.gradient_checkpointing_kwargs = dict(use_reentrant=args.gradient_checkpointing_use_reentrant)
    if args.test is True:
        args.report_to = []

    print(args)
    print(model_config)

    # Seed
    if args.full_determinism is True:
        enable_full_determinism(args.seed)
    else:
        set_seed(args.seed)

    # Model init kwargs & Tokenizer
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({
            'pad_token': '<|pad|>'
        })
        model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Dataset
    train_dataset_list = []
    for dataset_path in args.dataset_path:
        train_dataset = load_dataset("json", data_files=dataset_path, split='train')
        columns_to_keep = ['prompt', 'completion']
        columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]
        train_dataset = train_dataset.remove_columns(columns_to_remove)
        train_dataset_list.append(train_dataset)
    train_dataset = concatenate_datasets(train_dataset_list)

    if args.dataset_size > 0:
        train_dataset = train_dataset.select(range(args.dataset_size))
    if args.test is True:
        train_dataset = train_dataset.select(range(1024))

    padding = True
    max_length = None
    if args.test is True:
        padding = 'max_length'
        max_length = args.max_seq_length
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=get_response_template_ids(model_config.model_name_or_path, tokenizer, args.response_template),
        padding=padding,
        max_length=max_length
    )

    # Training
    trainer = SFTTrainer(
        args=args,
        model=model,
        peft_config=get_peft_config(model_config),
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    if args.save_strategy == 'no':
        trainer.save_model(args.output_dir)
