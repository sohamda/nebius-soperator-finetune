"""
Mistral-7B JSON-Mode Fine-tuning Script with Validation
Trains the model with validation metrics and early stopping.

Usage:
    python train_with_eval_mistral.py [--train-data PATH] [--val-data PATH] [--output-dir PATH]
"""

import os
import argparse
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import math

# Default paths
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
DEFAULT_TRAIN_DATA = "/mnt/shared/data/train.jsonl"
DEFAULT_VAL_DATA = "/mnt/shared/data/val.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/shared/checkpoints/mistral-json-mode"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B for JSON-only output with validation")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=str, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=str, default=DEFAULT_VAL_DATA)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=50)
    return parser.parse_args()


# System prompt used during training — MUST match inference
JSON_SYSTEM_PROMPT = (
    "You are a helpful assistant that always responds with a valid JSON object. "
    "Your response must be ONLY a JSON object with no other text."
)


def format_prompt(example):
    """Format dataset example for training with Mistral's chat format."""
    messages = example.get("messages", [])
    
    # Build conversation using Mistral's [INST] format
    # First message should include system prompt if present
    text = ""
    
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            if i == 0:
                # First user message - include system prompt
                text += f"<s>[INST] <<SYS>>\n{JSON_SYSTEM_PROMPT}\n<</SYS>>\n\n{content} [/INST]"
            else:
                # Subsequent user messages
                text += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            # Assistant response
            text += f" {content}</s>"
    
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for causal LM training."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    result["labels"] = result["input_ids"].clone()
    return result


def main():
    args = parse_args()
    
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        print("=" * 70)
        print("Mistral-7B JSON-Mode Fine-tuning with Validation")
        print("=" * 70)
        print(f"Model: {args.model_name}")
        print(f"Training data: {args.train_data}")
        print(f"Validation data: {args.val_data}")
        print(f"Output directory: {args.output_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max sequence length: {args.max_seq_length}")
        print(f"LoRA r: {args.lora_r}")
        print(f"LoRA alpha: {args.lora_alpha}")
        print(f"Use 4-bit: {args.use_4bit}")
        print(f"Early stopping patience: {args.early_stopping_patience}")
        print(f"Evaluation steps: {args.eval_steps}")
        print(f"World size: {world_size}")
        print("=" * 70)

    # Load tokenizer
    if local_rank == 0:
        print(f"\n📦 Loading tokenizer... (local_rank={local_rank})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    if local_rank == 0:
        print(f"📦 Loading model... (local_rank={local_rank})")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": local_rank},
    }
    
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        attn_implementation="eager",  # Flash Attention 2 not available, fallback to eager
        **model_kwargs,
    )

    # Configure LoRA
    if local_rank == 0:
        print("🔧 Configuring LoRA...")
    
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()

    # Load datasets
    if local_rank == 0:
        print("\n📂 Loading datasets...")
    
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    val_dataset = load_dataset("json", data_files=args.val_data, split="train")
    
    if local_rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    # Format datasets
    if local_rank == 0:
        print("🔄 Formatting datasets with Mistral chat format...")
    
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_prompt, remove_columns=val_dataset.column_names)

    # Tokenize datasets
    if local_rank == 0:
        print("🔄 Tokenizing datasets...")
    
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # Training arguments
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.001,
    )

    # Trainer
    if local_rank == 0:
        print("\n🚀 Starting training...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    # Train
    trainer.train()

    # Save final model
    if local_rank == 0:
        print("\n💾 Saving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        metrics = trainer.state.log_history
        with open(f"{output_dir}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
