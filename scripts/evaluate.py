"""
Model Evaluation Script
Evaluates fine-tuned model on test set and reports metrics.

Usage:
    python evaluate.py --model-path PATH --test-data PATH
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    total_examples: int = 0
    successful_generations: int = 0
    json_valid: int = 0
    json_has_answer: int = 0
    json_has_type: int = 0
    json_has_confidence: int = 0
    average_response_length: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def generation_rate(self) -> float:
        return self.successful_generations / self.total_examples if self.total_examples > 0 else 0
    
    @property
    def json_validity_rate(self) -> float:
        return self.json_valid / self.successful_generations if self.successful_generations > 0 else 0
    
    @property
    def schema_compliance_rate(self) -> float:
        """Rate of JSONs with all expected fields."""
        if self.json_valid == 0:
            return 0
        return min(self.json_has_answer, self.json_has_type, self.json_has_confidence) / self.json_valid
    
    def to_dict(self) -> Dict:
        return {
            "total_examples": self.total_examples,
            "successful_generations": self.successful_generations,
            "generation_rate": f"{self.generation_rate:.2%}",
            "json_valid": self.json_valid,
            "json_validity_rate": f"{self.json_validity_rate:.2%}",
            "json_has_answer": self.json_has_answer,
            "json_has_type": self.json_has_type,
            "json_has_confidence": self.json_has_confidence,
            "schema_compliance_rate": f"{self.schema_compliance_rate:.2%}",
            "average_response_length": f"{self.average_response_length:.1f}",
            "error_count": len(self.errors),
        }


def load_model(
    base_model: str,
    adapter_path: Optional[str] = None,
    device: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with optional LoRA adapter."""
    print(f"Loading base model: {base_model}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_test_data(filepath: str) -> List[Dict]:
    """Load test dataset from JSONL file."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def extract_prompt(example: Dict) -> str:
    """Extract user prompt from example."""
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_expected(example: Dict) -> Optional[Dict]:
    """Extract expected JSON response from example."""
    messages = example.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                return json.loads(msg.get("content", "{}"))
            except json.JSONDecodeError:
                return None
    return None


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate model response for a prompt."""
    # Format prompt for Phi-3
    formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response


def validate_json_response(response: str) -> Tuple[bool, Optional[Dict]]:
    """Check if response is valid JSON and extract it."""
    response = response.strip()
    
    # Try to find JSON in the response
    start = response.find("{")
    end = response.rfind("}") + 1
    
    if start != -1 and end > start:
        json_str = response[start:end]
        try:
            parsed = json.loads(json_str)
            return True, parsed
        except json.JSONDecodeError:
            pass
    
    return False, None


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: List[Dict],
    max_examples: Optional[int] = None,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate model on test dataset."""
    result = EvaluationResult()
    
    if max_examples:
        test_data = test_data[:max_examples]
    
    result.total_examples = len(test_data)
    total_length = 0
    
    for i, example in enumerate(test_data):
        prompt = extract_prompt(example)
        expected = extract_expected(example)
        
        if not prompt:
            result.errors.append(f"Example {i}: No prompt found")
            continue
        
        try:
            response = generate_response(model, tokenizer, prompt)
            result.successful_generations += 1
            total_length += len(response)
            
            is_valid, parsed = validate_json_response(response)
            
            if is_valid:
                result.json_valid += 1
                
                if "answer" in parsed:
                    result.json_has_answer += 1
                if "type" in parsed:
                    result.json_has_type += 1
                if "confidence" in parsed:
                    result.json_has_confidence += 1
            
            if verbose:
                print(f"\n--- Example {i+1}/{len(test_data)} ---")
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print(f"Valid JSON: {is_valid}")
                if is_valid and parsed:
                    print(f"Fields: {list(parsed.keys())}")
        
        except Exception as e:
            result.errors.append(f"Example {i}: {str(e)}")
    
    if result.successful_generations > 0:
        result.average_response_length = total_length / result.successful_generations
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test.jsonl",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each example",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum JSON validity rate to pass (default: 0.8)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Adapter: {args.adapter_path or 'None'}")
    print(f"Test data: {args.test_data}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model(args.model_path, args.adapter_path)
    
    # Load test data
    if not os.path.exists(args.test_data):
        print(f"ERROR: Test data not found: {args.test_data}")
        sys.exit(1)
    
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate
    print("\nRunning evaluation...")
    result = evaluate_model(
        model, tokenizer, test_data,
        max_examples=args.max_examples,
        verbose=args.verbose,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, value in result.to_dict().items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model_path,
            "adapter": args.adapter_path,
            "test_data": args.test_data,
            "results": result.to_dict(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Check threshold
    if result.json_validity_rate >= args.threshold:
        print(f"\n✅ PASS: JSON validity rate {result.json_validity_rate:.1%} >= {args.threshold:.1%}")
        return 0
    else:
        print(f"\n❌ FAIL: JSON validity rate {result.json_validity_rate:.1%} < {args.threshold:.1%}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
