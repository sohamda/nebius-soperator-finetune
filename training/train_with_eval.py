"""
Phi-3 Mini JSON-Mode Fine-tuning Script with Validation
Trains the model with validation metrics and early stopping.

Usage:
    python train_with_eval.py [--train-data PATH] [--val-data PATH] [--output-dir PATH]
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
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_TRAIN_DATA = "/mnt/shared/data/train.jsonl"
DEFAULT_VAL_DATA = "/mnt/shared/data/val.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/shared/checkpoints/phi3-json-mode"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 for JSON-only output with validation")
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
    """Format dataset example for training with system prompt."""
    messages = example.get("messages", [])
    # Always prepend system prompt so model learns JSON output with this context
    text = f"<|system|>\n{JSON_SYSTEM_PROMPT}<|end|>\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}<|end|>\n"
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for causal LM training."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Phi-3 JSON-Mode Fine-tuning with Validation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1
    
    # Quantization config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    print(f"\n📦 Loading model... (local_rank={local_rank})")
    # Use Flash Attention 2 for better GPU utilization (falls back to eager if not available)
    try:
        attn_impl = "flash_attention_2"
        print("Using Flash Attention 2 for optimized GPU performance")
    except:
        attn_impl = "eager"
        print("Flash Attention 2 not available, using eager attention")
    
    if is_distributed:
        # DDP: each process loads model on its own GPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation=attn_impl,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    print("\n🔧 Trainable parameters:")
    model.print_trainable_parameters()
    
    # Load datasets
    print(f"\n📊 Loading training data from {args.train_data}...")
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    print(f"Training examples: {len(train_dataset)}")
    
    print(f"📊 Loading validation data from {args.val_data}...")
    val_dataset = load_dataset("json", data_files=args.val_data, split="train")
    val_dataset = val_dataset.map(format_prompt, remove_columns=val_dataset.column_names)
    print(f"Validation examples: {len(val_dataset)}")
    
    # Tokenize datasets
    print("\n🔤 Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments with evaluation
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.eval_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        report_to="none",
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        # Optimized dataloader settings for maximum GPU utilization
        dataloader_num_workers=8,           # Increased from 4
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,       # Prefetch 2 batches ahead
        dataloader_persistent_workers=True, # Keep workers alive between epochs
    )
    
    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Train
    print("\n🚀 Starting training...")
    train_result = trainer.train()
    
    # Save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    print("\n📈 Running final evaluation...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(val_dataset)
    
    # Compute perplexity
    eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
    
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n📊 Final Metrics:")
    print(f"  Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}")
    print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
    
    # Save
    print(f"\n💾 Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training summary
    summary = {
        "model": args.model_name,
        "train_data": args.train_data,
        "val_data": args.val_data,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "epochs": args.epochs,
        "final_train_loss": metrics.get("train_loss"),
        "final_eval_loss": eval_metrics["eval_loss"],
        "perplexity": eval_metrics["perplexity"],
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
