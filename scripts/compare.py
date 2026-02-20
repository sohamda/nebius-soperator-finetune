"""
Model Comparison Script
Compares responses from base Phi-3 vs JSON-fine-tuned Phi-3

Usage:
    python compare.py [--base-url URL] [--json-url URL]
"""

import argparse
import json
import requests
from typing import Optional


# Default endpoints (adjust based on your setup)
DEFAULT_BASE_URL = "http://localhost:8001/v1/chat/completions"
DEFAULT_JSON_URL = "http://localhost:8000/v1/chat/completions"

# Test prompts
TEST_PROMPTS = [
    "What is the capital of Japan?",
    "Explain quantum computing in simple terms",
    "What is 15 * 23?",
    "Who wrote Romeo and Juliet?",
    "What are the primary colors?",
    "How does photosynthesis work?",
    "What is the speed of light?",
    "Name three programming languages",
]


def query_model(url: str, model_name: str, prompt: str) -> Optional[str]:
    """Send a chat completion request to the model."""
    try:
        response = requests.post(
            url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.7,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"[ERROR: {e}]"


def validate_json(response: str) -> tuple[bool, Optional[dict]]:
    """Check if the response is valid JSON."""
    try:
        # Try to find JSON in the response
        response = response.strip()
        
        # Handle responses that might have text before/after JSON
        start = response.find("{")
        end = response.rfind("}") + 1
        
        if start != -1 and end > start:
            json_str = response[start:end]
            parsed = json.loads(json_str)
            return True, parsed
        return False, None
    except json.JSONDecodeError:
        return False, None


def format_response(response: str, max_width: int = 70) -> str:
    """Format response for display."""
    lines = []
    current_line = ""
    
    for word in response.split():
        if len(current_line) + len(word) + 1 <= max_width:
            current_line += (" " if current_line else "") + word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return "\n    ".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs JSON-finetuned model")
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base model endpoint URL",
    )
    parser.add_argument(
        "--json-url",
        type=str,
        default=DEFAULT_JSON_URL,
        help="JSON-finetuned model endpoint URL",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="phi3-base",
        help="Base model name",
    )
    parser.add_argument(
        "--json-model",
        type=str,
        default="phi3-json",
        help="JSON model name",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MODEL COMPARISON: Base Phi-3 vs JSON-Fine-tuned Phi-3")
    print("=" * 80)
    print(f"Base Model URL:  {args.base_url}")
    print(f"JSON Model URL:  {args.json_url}")
    print("=" * 80)

    results = {
        "base_success": 0,
        "json_success": 0,
        "json_valid": 0,
        "total": len(TEST_PROMPTS),
    }

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 80}")
        print(f"PROMPT {i}/{len(TEST_PROMPTS)}: {prompt}")
        print("─" * 80)

        # Query base model
        print("\n🔵 BASE MODEL:")
        base_response = query_model(args.base_url, args.base_model, prompt)
        if not base_response.startswith("[ERROR"):
            results["base_success"] += 1
        print(f"    {format_response(base_response)}")

        # Query JSON model
        print("\n🟢 JSON MODEL:")
        json_response = query_model(args.json_url, args.json_model, prompt)
        if not json_response.startswith("[ERROR"):
            results["json_success"] += 1
        print(f"    {format_response(json_response)}")

        # Validate JSON
        is_valid, parsed = validate_json(json_response)
        if is_valid:
            results["json_valid"] += 1
            print("\n    ✅ Valid JSON!")
            if parsed:
                print(f"    Fields: {list(parsed.keys())}")
        else:
            print("\n    ❌ Invalid JSON")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total prompts:        {results['total']}")
    print(f"Base model responses: {results['base_success']}/{results['total']}")
    print(f"JSON model responses: {results['json_success']}/{results['total']}")
    print(f"Valid JSON outputs:   {results['json_valid']}/{results['total']} "
          f"({100 * results['json_valid'] / results['total']:.1f}%)")
    print("=" * 80)

    # Return exit code based on JSON validity rate
    json_rate = results["json_valid"] / results["total"] if results["total"] > 0 else 0
    if json_rate >= 0.8:
        print("\n✅ SUCCESS: JSON model producing valid JSON ≥80% of the time")
        return 0
    else:
        print("\n⚠️  WARNING: JSON model validity rate below 80%")
        return 1


if __name__ == "__main__":
    exit(main())
