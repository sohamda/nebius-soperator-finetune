"""
Inference Comparison: Base Phi-3 vs JSON-Fine-tuned Phi-3
Loads both models on a single GPU and compares outputs.

Modes:
  --mode different-prompt  (default) Base=no system prompt, Fine-tuned=JSON system prompt
  --mode same-prompt       Both models get the same JSON system prompt

Usage:
    python run_inference_compare.py --mode different-prompt --output-tag diff
    python run_inference_compare.py --mode same-prompt --output-tag same
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_CHECKPOINT = "/mnt/shared/checkpoints/phi3-json-mode"
DEFAULT_TEST_DATA = "/mnt/shared/data/test.jsonl"
DEFAULT_OUTPUT_DIR = "/mnt/shared/logs"

# Test prompts for comparison
TEST_PROMPTS = [
    "What is the capital of Japan?",
    "Explain quantum computing in simple terms",
    "What is 15 * 23?",
    "Who wrote Romeo and Juliet?",
    "What are the primary colors?",
    "How does photosynthesis work?",
    "What is the speed of light?",
    "Name three programming languages",
    "What is DNA?",
    "When was the moon landing?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned Phi-3")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-data", type=str, default=DEFAULT_TEST_DATA)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mode", type=str, default="no-system",
                        choices=["no-system", "with-system"],
                        help="no-system: both models get NO sys prompt. with-system: both get JSON sys prompt.")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Tag for output filename, e.g. 'nosys' or 'withsys'")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_base_model(model_name, use_4bit):
    """Load the base (un-finetuned) model."""
    print(f"\n{'='*70}")
    print(f"Loading BASE model: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", **kwargs
    )
    model.eval()
    print(f"Base model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


def load_finetuned_model(model_name, checkpoint_dir, use_4bit):
    """Load the base model + LoRA adapter (fine-tuned)."""
    print(f"\n{'='*70}")
    print(f"Loading FINE-TUNED model: {model_name} + LoRA from {checkpoint_dir}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", **kwargs
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()
    print(f"Fine-tuned model loaded. LoRA adapter from: {checkpoint_dir}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256, use_json_system=False):
    """Generate a response from the model.
    
    Uses the exact same chat template format as training:
      <|user|>\n{content}<|end|>\n<|assistant|>\n
    """
    if use_json_system:
        # Add system message to encourage JSON output
        input_text = (
            "<|system|>\nYou are a helpful assistant that always responds with a valid JSON object. "
            "Your response must be ONLY a JSON object with no other text.<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            "<|assistant|>\n"
        )
    else:
        # Match training format exactly
        input_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic comparison
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the input)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def validate_json(response):
    """Check if the response is valid JSON."""
    try:
        response = response.strip()
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            return True, parsed
        return False, None
    except json.JSONDecodeError:
        return False, None


def run_comparison(args):
    """Run full comparison between base and fine-tuned models."""

    mode = args.mode
    tag = args.output_tag or mode
    use_sys = (mode == "with-system")  # both models get the same treatment

    print(f"\n{'#'*70}")
    print(f"# MODE: {mode}")
    print(f"#   System prompt for BOTH models: {'JSON' if use_sys else 'NONE'}")
    print(f"{'#'*70}")

    # ── Phase 1: Base model inference ──
    base_model, base_tokenizer = load_base_model(args.model_name, args.use_4bit)

    base_results = []
    print(f"\n{'='*70}")
    print(f"PHASE 1: BASE MODEL INFERENCE (system prompt: {'JSON' if use_sys else 'NONE'})")
    print(f"{'='*70}")
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
        t0 = time.time()
        response = generate_response(base_model, base_tokenizer, prompt, args.max_new_tokens, use_json_system=use_sys)
        elapsed = time.time() - t0
        is_json, parsed = validate_json(response)
        base_results.append({
            "prompt": prompt,
            "response": response,
            "is_json": is_json,
            "time": elapsed,
        })
        print(f"  Response ({elapsed:.1f}s): {response[:200]}")
        if is_json:
            print(f"  ✅ Valid JSON: {list(parsed.keys())}")
        else:
            print(f"  ❌ Not JSON")

    # Free base model memory
    del base_model
    torch.cuda.empty_cache()

    # ── Phase 2: Fine-tuned model inference ──
    ft_model, ft_tokenizer = load_finetuned_model(
        args.model_name, args.checkpoint_dir, args.use_4bit
    )

    ft_results = []
    print(f"\n{'='*70}")
    print(f"PHASE 2: FINE-TUNED MODEL INFERENCE (system prompt: {'JSON' if use_sys else 'NONE'})")
    print(f"{'='*70}")
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
        t0 = time.time()
        response = generate_response(ft_model, ft_tokenizer, prompt, args.max_new_tokens, use_json_system=use_sys)
        elapsed = time.time() - t0
        is_json, parsed = validate_json(response)
        ft_results.append({
            "prompt": prompt,
            "response": response,
            "is_json": is_json,
            "time": elapsed,
        })
        print(f"  Response ({elapsed:.1f}s): {response[:200]}")
        if is_json:
            print(f"  ✅ Valid JSON: {list(parsed.keys())}")
        else:
            print(f"  ❌ Not JSON")

    del ft_model
    torch.cuda.empty_cache()

    # ── Phase 3: Summary ──
    base_json_count = sum(1 for r in base_results if r["is_json"])
    ft_json_count = sum(1 for r in ft_results if r["is_json"])
    total = len(TEST_PROMPTS)

    base_avg_time = sum(r["time"] for r in base_results) / total
    ft_avg_time = sum(r["time"] for r in ft_results) / total

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Base Model':<20} {'Fine-tuned':<20}")
    print(f"{'-'*70}")
    print(f"{'Valid JSON outputs':<30} {f'{base_json_count}/{total}':<20} {f'{ft_json_count}/{total}':<20}")
    print(f"{'JSON rate':<30} {f'{100*base_json_count/total:.0f}%':<20} {f'{100*ft_json_count/total:.0f}%':<20}")
    print(f"{'Avg response time':<30} {f'{base_avg_time:.1f}s':<20} {f'{ft_avg_time:.1f}s':<20}")
    print(f"{'='*70}")

    # Side-by-side comparison for each prompt
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE RESPONSES")
    print(f"{'='*70}")
    for i, (br, fr) in enumerate(zip(base_results, ft_results)):
        print(f"\n{'─'*70}")
        print(f"PROMPT: {br['prompt']}")
        print(f"{'─'*70}")
        print(f"  BASE:      {br['response'][:300]}")
        print(f"  FINETUNED: {fr['response'][:300]}")
        json_indicator_base = "✅" if br["is_json"] else "❌"
        json_indicator_ft = "✅" if fr["is_json"] else "❌"
        print(f"  JSON:      base={json_indicator_base}  finetuned={json_indicator_ft}")

    print(f"\n{'='*70}")
    ft_rate = ft_json_count / total if total > 0 else 0
    if ft_rate >= 0.8:
        print(f"✅ SUCCESS: Fine-tuned model produces valid JSON {100*ft_rate:.0f}% of the time (≥80% threshold)")
    else:
        print(f"⚠️  WARNING: Fine-tuned model JSON rate {100*ft_rate:.0f}% is below 80% threshold")
    print(f"{'='*70}")

    # Save results to file
    output_file = os.path.join(args.output_dir, f"comparison_{tag}.json")
    with open(output_file, "w") as f:
        json.dump({
            "mode": mode,
            "system_prompt_for_both": "JSON" if use_sys else "NONE",
            "base_results": base_results,
            "finetuned_results": ft_results,
            "summary": {
                "base_json_rate": base_json_count / total,
                "finetuned_json_rate": ft_json_count / total,
                "base_avg_time": base_avg_time,
                "finetuned_avg_time": ft_avg_time,
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return 0 if ft_rate >= 0.8 else 1


def main():
    args = parse_args()
    return run_comparison(args)


if __name__ == "__main__":
    exit(main())
