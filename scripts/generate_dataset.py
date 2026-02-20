"""
JSON Mode Dataset Generator
Generates training data for teaching Phi-3 to respond in JSON format.
Includes train/validation/test splits.

Usage:
    python generate_dataset.py [--output-dir PATH] [--count N] [--train-ratio 0.8] [--val-ratio 0.1]
"""

import argparse
import json
import random
import os
from typing import List, Dict, Any, Tuple


# Template questions and JSON responses
TEMPLATES = [
    # Geography
    {
        "category": "geography",
        "questions": [
            "What is the capital of {country}?",
            "Where is {country} located?",
            "What continent is {country} in?",
        ],
        "data": [
            {"country": "France", "capital": "Paris", "continent": "Europe"},
            {"country": "Japan", "capital": "Tokyo", "continent": "Asia"},
            {"country": "Brazil", "capital": "Brasília", "continent": "South America"},
            {"country": "Australia", "capital": "Canberra", "continent": "Australia"},
            {"country": "Egypt", "capital": "Cairo", "continent": "Africa"},
            {"country": "Canada", "capital": "Ottawa", "continent": "North America"},
            {"country": "Germany", "capital": "Berlin", "continent": "Europe"},
            {"country": "India", "capital": "New Delhi", "continent": "Asia"},
            {"country": "Mexico", "capital": "Mexico City", "continent": "North America"},
            {"country": "South Africa", "capital": "Pretoria", "continent": "Africa"},
        ],
    },
    # Math
    {
        "category": "math",
        "questions": [
            "What is {a} + {b}?",
            "Calculate {a} * {b}",
            "What is {a} - {b}?",
            "What is {a} divided by {b}?",
        ],
    },
    # Science
    {
        "category": "science",
        "questions": [
            "What is {concept}?",
            "Explain {concept}",
            "Define {concept}",
        ],
        "data": [
            {"concept": "photosynthesis", "definition": "Process by which plants convert sunlight, water, and CO2 into glucose and oxygen"},
            {"concept": "gravity", "definition": "Fundamental force of attraction between objects with mass"},
            {"concept": "DNA", "definition": "Deoxyribonucleic acid, molecule that carries genetic instructions"},
            {"concept": "evolution", "definition": "Process of gradual change in species over generations through natural selection"},
            {"concept": "mitosis", "definition": "Cell division process that produces two identical daughter cells"},
            {"concept": "entropy", "definition": "Measure of disorder or randomness in a system"},
            {"concept": "ecosystem", "definition": "Community of organisms interacting with their physical environment"},
            {"concept": "atom", "definition": "Smallest unit of matter that retains chemical properties of an element"},
        ],
    },
    # General knowledge
    {
        "category": "general",
        "questions": [
            "Who wrote {work}?",
            "When was {event}?",
            "What is {thing}?",
        ],
        "data": [
            {"work": "Romeo and Juliet", "author": "William Shakespeare", "year": "1597"},
            {"work": "1984", "author": "George Orwell", "year": "1949"},
            {"work": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": "1925"},
            {"event": "World War II ended", "year": "1945"},
            {"event": "the moon landing", "year": "1969"},
            {"event": "the French Revolution", "year": "1789"},
            {"thing": "the speed of light", "value": "299,792,458 m/s", "type": "physics constant"},
            {"thing": "water", "formula": "H2O", "type": "chemical compound"},
        ],
    },
    # Programming
    {
        "category": "programming",
        "questions": [
            "What is {language} used for?",
            "Name a feature of {language}",
            "Describe {concept}",
        ],
        "data": [
            {"language": "Python", "use": "general purpose programming, data science, web development", "feature": "dynamic typing"},
            {"language": "JavaScript", "use": "web development, frontend and backend", "feature": "event-driven programming"},
            {"language": "Rust", "use": "systems programming, performance-critical applications", "feature": "memory safety without garbage collection"},
            {"language": "Go", "use": "cloud services, networking, concurrent applications", "feature": "built-in concurrency with goroutines"},
            {"concept": "recursion", "definition": "programming technique where a function calls itself"},
            {"concept": "API", "definition": "Application Programming Interface, set of rules for software interaction"},
        ],
    },
]


def generate_math_example():
    """Generate a math question with JSON answer."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    
    ops = [
        ("+", a + b, "addition"),
        ("*", a * b, "multiplication"),
        ("-", a - b, "subtraction"),
    ]
    
    # Add division only if cleanly divisible
    if b != 0 and a % b == 0:
        ops.append(("/", a // b, "division"))
    
    op, result, op_type = random.choice(ops)
    
    questions = [
        f"What is {a} {op} {b}?",
        f"Calculate {a} {op} {b}",
        f"Compute: {a} {op} {b}",
    ]
    
    return {
        "prompt": random.choice(questions),
        "response": {
            "answer": result,
            "type": "math",
            "operation": op_type,
            "confidence": 1.0,
        },
    }


def generate_example(template):
    """Generate a single training example from a template."""
    category = template["category"]
    
    if category == "math":
        return generate_math_example()
    
    data = template.get("data", [])
    if not data:
        return None
    
    item = random.choice(data)
    question_template = random.choice(template["questions"])
    
    # Fill in the template
    try:
        question = question_template.format(**item)
    except KeyError:
        return None
    
    # Build response based on category
    if category == "geography":
        response = {
            "answer": item.get("capital", item.get("continent", "Unknown")),
            "type": "geography",
            "confidence": 0.99,
        }
    elif category == "science":
        response = {
            "answer": item.get("definition", ""),
            "type": "science",
            "concept": item.get("concept", ""),
            "confidence": 0.95,
        }
    elif category == "general":
        if "author" in item:
            response = {
                "answer": item["author"],
                "type": "literature",
                "work": item.get("work", ""),
                "year": item.get("year", ""),
                "confidence": 0.98,
            }
        elif "year" in item:
            response = {
                "answer": item["year"],
                "type": "history",
                "event": item.get("event", ""),
                "confidence": 0.97,
            }
        else:
            response = {
                "answer": item.get("value", item.get("formula", "")),
                "type": item.get("type", "general"),
                "confidence": 0.96,
            }
    elif category == "programming":
        if "language" in item:
            response = {
                "answer": item.get("use", item.get("feature", "")),
                "type": "programming",
                "language": item["language"],
                "confidence": 0.94,
            }
        else:
            response = {
                "answer": item.get("definition", ""),
                "type": "programming",
                "concept": item.get("concept", ""),
                "confidence": 0.93,
            }
    else:
        response = {
            "answer": str(item),
            "type": category,
            "confidence": 0.90,
        }
    
    return {
        "prompt": question,
        "response": response,
    }


def format_for_training(example: Dict) -> Dict:
    """Format example for SFT training."""
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": json.dumps(example["response"])},
        ]
    }


def split_dataset(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/validation/test sets."""
    random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_set = examples[:train_end]
    val_set = examples[train_end:val_end]
    test_set = examples[val_end:]
    
    return train_set, val_set, test_set


def write_jsonl(examples: List[Dict], filepath: str) -> None:
    """Write examples to JSONL file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate JSON mode training dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Total number of examples to generate",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction for training set (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction for validation set (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    
    print(f"Generating {args.count} training examples...")
    print(f"Split: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / {1-args.train_ratio-args.val_ratio:.0%} test")
    
    # Generate examples
    examples = []
    attempts = 0
    max_attempts = args.count * 10
    
    while len(examples) < args.count and attempts < max_attempts:
        attempts += 1
        template = random.choice(TEMPLATES)
        example = generate_example(template)
        
        if example:
            examples.append(format_for_training(example))
    
    # Split dataset
    train_set, val_set, test_set = split_dataset(
        examples, args.train_ratio, args.val_ratio
    )
    
    # Write files
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    write_jsonl(train_set, train_path)
    write_jsonl(val_set, val_path)
    write_jsonl(test_set, test_path)
    
    # Also write combined file for backward compatibility
    combined_path = os.path.join(args.output_dir, "json_mode_dataset.jsonl")
    write_jsonl(examples, combined_path)
    
    print(f"\n✅ Generated {len(examples)} examples")
    print(f"📊 Split:")
    print(f"   Train: {len(train_set)} examples → {train_path}")
    print(f"   Val:   {len(val_set)} examples → {val_path}")
    print(f"   Test:  {len(test_set)} examples → {test_path}")
    print(f"   All:   {len(examples)} examples → {combined_path}")
    
    # Show sample
    print("\n📝 Sample example:")
    sample = random.choice(examples)
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
