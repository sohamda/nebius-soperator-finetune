"""
Inference Comparison: Base Mistral-7B vs JSON-Fine-tuned Mistral-7B
Loads both models on a single GPU and compares outputs.

Modes:
  --mode no-system    Both models get NO system prompt
  --mode with-system  Both models get the same JSON system prompt

Usage:
    python run_inference_compare_mistral.py --mode no-system --output-tag nosys
    python run_inference_compare_mistral.py --mode with-system --output-tag withsys
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
DEFAULT_CHECKPOINT = "/mnt/shared/checkpoints/mistral-7b-json-mode"
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
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned Mistral-7B")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-data", type=str, default=DEFAULT_TEST_DATA)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=150)
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
    """Generate a response from the model using Mistral's [INST] format.
    
    Mistral chat format:
      - Without system: <s>[INST] {prompt} [/INST]
      - With system:    <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]
    """
    if use_json_system:
        # Add system message to encourage JSON output
        input_text = (
            "<s>[INST] <<SYS>>\n"
            "You are a helpful assistant that always responds with a valid JSON object. "
            "Your response must be ONLY a JSON object with no other text.\n"
            "<</SYS>>\n\n"
            f"{prompt} [/INST]"
        )
    else:
        # No system prompt - just the instruction
        input_text = f"<s>[INST] {prompt} [/INST]"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic comparison
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip the input)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Clean up: remove everything after [/INST] or [INST] if present
    for stop_marker in ["[/INST]", "[INST]", "<</SYS>>"]:
        if stop_marker in response:
            response = response[:response.index(stop_marker)].strip()
    
    return response


def validate_json(response):
    """Check if the response is valid JSON."""
    try:
        response = response.strip()
        
        # Find first complete JSON object
        start = response.find("{")
        if start == -1:
            return False, None
        
        # Find matching closing brace
        depth = 0
        end = -1
        for i in range(start, len(response)):
            if response[i] == '{':
                depth += 1
            elif response[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        if end == -1:
            return False, None
        
        # Try to parse just the first JSON object
        json_str = response[start:end]
        parsed = json.loads(json_str)
        return True, parsed
    except (json.JSONDecodeError, ValueError):
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
    print("MISTRAL-7B COMPARISON SUMMARY")
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
    output_file = os.path.join(args.output_dir, f"comparison_mistral_{tag}.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": "mistralai/Mistral-7B-v0.1",
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
