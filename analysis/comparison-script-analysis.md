# Comparison Script Analysis: Base vs Fine-tuned Model Inference

## 🎯 Overview

**Script**: `scripts/run_inference_compare_mistral.py`  
**Supporting Scripts**: `scripts/evaluate.py`, `scripts/compare.py`  
**Slurm Job**: `slurm/compare_mistral.sh`  
**Purpose**: Compare base Mistral-7B vs fine-tuned model on JSON generation task  
**Strategy**: Two-mode testing (no-system vs with-system prompts)  
**Key Result**: Fine-tuned achieves **90% JSON validity** vs **0% for base model**

---

## 📋 Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Test Dataset (data/test.jsonl)                              │
│  80 diverse samples (10% of total dataset)                   │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  run_inference_compare_mistral.py                            │
│  ├─ Two comparison modes:                                    │
│  │  1. No system prompt (pure model capability)             │
│  │  2. With system prompt (explicit JSON instruction)        │
│  └─ Sequential model loading (base → fine-tuned)            │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Mode 1: No System Prompt                                    │
│  ├─ Load base model (4-bit quantized)                       │
│  ├─ Generate predictions for 80 samples                      │
│  ├─ Unload base model (free memory)                         │
│  ├─ Load fine-tuned model (4-bit quantized)                 │
│  ├─ Generate predictions for 80 samples                      │
│  └─ Save results → comparison_mistral_nosys.json            │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Mode 2: With System Prompt                                  │
│  ├─ Add "Output in valid JSON format" instruction           │
│  ├─ Run same pipeline as Mode 1                             │
│  └─ Save results → comparison_mistral_withsys.json          │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  evaluate.py                                                 │
│  ├─ JSON validity check (brace matching)                    │
│  ├─ Calculate validity rate (valid/total)                   │
│  └─ Report metrics for each mode                            │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  compare.py                                                  │
│  ├─ Aggregate results across modes                          │
│  ├─ Compare base vs fine-tuned performance                   │
│  └─ Save combined results → comparison_mistral_combined.json│
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Script Deep Dive: `run_inference_compare_mistral.py`

### Key Design Decisions

#### 1. Sequential Model Loading (Not Parallel)

**Current Implementation**:
```python
# Load base model
base_model = load_model(base_model_path)
base_predictions = generate(base_model, test_data)
del base_model  # Explicitly free memory
torch.cuda.empty_cache()

# Load fine-tuned model
finetuned_model = load_model(checkpoint_path)
finetuned_predictions = generate(finetuned_model, test_data)
```

**Why Sequential?**
- **Memory Constraint**: Each model needs ~18-20GB even with 4-bit quantization
- **Single GPU**: H100 has 80GB, but keeping both models would leave no room for activations
- **Safety**: Prevents OOM that would waste entire inference job

**Technical Explanation**: 
> "I load models sequentially to avoid memory issues. Even with 4-bit quantization, each 7B model needs ~18GB. Loading both simultaneously would risk OOM, so I process base model, unload it, then process fine-tuned model. This doubles inference time but guarantees successful runs."

**Trade-off Analysis**:

| Approach | Memory | Time | Reliability |
|----------|--------|------|-------------|
| **Sequential** | 20GB peak | 2× longer | ✅ Always works |
| **Parallel** | 40GB peak | 1× baseline | ⚠️ Risk OOM |
| **Multi-GPU** | 20GB per GPU | 1× baseline | ✅ Best of both (if available) |

---

#### 2. 4-bit Quantization for Inference

**Implementation**:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 4-bit quantization
    bnb_4bit_use_double_quant=True,      # Double quantization (extra compression)
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (better for neural nets)
    bnb_4bit_compute_dtype=torch.bfloat16  # Compute in BF16 (accuracy)
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

**Why 4-bit Quantization?**

| Precision | Memory | Speed | Accuracy | Use Case |
|-----------|--------|-------|----------|----------|
| **FP32** | 28GB | 1x | 100% | Not practical |
| **BF16** | 14GB | 2x | 99.9% | Training |
| **8-bit** | 7GB | 1.8x | 99% | Balanced |
| **4-bit** | 3.5GB | 1.5x | 97% | Inference |

**Your Choice: 4-bit NF4**
- **Memory**: Mistral-7B goes from 14GB → 3.5GB
- **Accuracy**: NF4 (NormalFloat4) preserves more info than standard 4-bit
- **Compute**: BF16 compute_dtype keeps activations high precision
- **Trade-off**: ~3% accuracy loss for 75% memory savings

**Technical Explanation**: 
> "I used 4-bit NF4 quantization to compress models from 14GB to 3.5GB with minimal quality loss. The double quantization and BF16 compute_dtype preserve accuracy while enabling both models to run on a single H100. This is a standard inference optimization for LLMs."

---

#### 3. Greedy Decoding (Not Sampling)

**Implementation**:
```python
generate_kwargs = {
    "max_new_tokens": 8192,        # Long output for JSON
    "do_sample": False,             # Greedy decoding (no randomness)
    "temperature": None,            # Not used in greedy
    "top_p": None,                  # Not used in greedy
    "top_k": None,                  # Not used in greedy
    "pad_token_id": tokenizer.eos_token_id,
}
```

**Why Greedy Decoding?**

| Strategy | Behavior | Reproducibility | Quality | Use Case |
|----------|----------|------------------|---------|----------|
| **Greedy** | Always pick highest prob token | ✅ Deterministic | Good | Testing, JSON |
| **Sampling (temp=0.7)** | Controlled randomness | ❌ Non-deterministic | Creative | Chat, stories |
| **Top-k** | Sample from top k tokens | ❌ Non-deterministic | Balanced | General |
| **Top-p (nucleus)** | Sample from cumulative p mass | ❌ Non-deterministic | High quality | Production |

**Your Choice: Greedy**
- **Reproducibility**: Same input = same output (critical for fair comparison)
- **Consistency**: No random variation between runs
- **JSON Task**: Structured output benefits from deterministic generation

**Technical Explanation**: 
> "I used greedy decoding without sampling to ensure reproducible comparisons. Since JSON requires strict structure, deterministic generation is more appropriate than randomness. This also makes debugging easier—same input always produces same output."

---

#### 4. Two-Mode Comparison (No-System vs With-System)

**Mode 1: No System Prompt** (Pure Model Capability)
```python
messages = [
    {"role": "user", "content": user_prompt}
]
```

**Mode 2: With System Prompt** (Explicit Instruction)
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Output in valid JSON format."},
    {"role": "user", "content": user_prompt}
]
```

**Why Test Both Modes?**

| Mode | Purpose | What It Tests | Expected Result |
|------|---------|---------------|-----------------|
| **No System** | Pure model capability | Has model learned JSON? | Fine-tuned should excel |
| **With System** | Instruction following | Can prompt rescue base model? | Shows fine-tuning value |

**Results Comparison**:

| Model | No System | With System | Improvement |
|-------|-----------|-------------|-------------|
| **Base Model** | 0% valid | ~10-20% valid | +20% |
| **Fine-tuned** | 85% valid | **90% valid** | +5% |

**Technical Explanation**: 
> "I tested two modes to isolate the fine-tuning's impact. The no-system mode tests what the model inherently learned, while with-system shows if prompting alone could solve the task. Results show fine-tuning is essential—base model achieves only 0-20% JSON validity even with instructions, while fine-tuned reaches 90%."

---

### Mistral Chat Format

**Implementation**:
```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Example output:
# <s>[INST] user message here [/INST]
```

**Why Chat Template?**
- **Mistral-specific**: Model was pretrained with this format
- **Instruction boundary**: `[INST]...[/INST]` signals where to respond
- **Better quality**: Using wrong format degrades performance

**Technical Explanation**: 
> "I used Mistral's official chat template which wraps inputs in `[INST]...[/INST]` tags. This matches the pretraining format, ensuring the model understands where instructions end and generation begins. Using raw prompts would degrade quality."

---

### Padding & Tokenization

**Implementation**:
```python
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # LEFT padding for causal LM

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=6144,
).to(device)
```

**Why Left Padding?**

| Padding Side | Behavior | Use Case |
|--------------|----------|----------|
| **Right** | Add padding after text | Encoder models (BERT) |
| **Left** | Add padding before text | Decoder models (GPT, Mistral) |

**Your Choice: Left**
- **Causal LM Requirement**: Generation starts from last real token
- **Batch Inference**: Ensures all sequences end at same position
- **Wrong padding**: Would generate from padding tokens → garbage output

**Technical Explanation**: 
> "I used left padding required for causal language models like Mistral. Right padding would place generation tokens in the middle of padding, producing garbage. Left padding ensures generation always starts from real text."

---

## 📊 Evaluation Pipeline

### `evaluate.py` - JSON Validity Check

**Implementation**:
```python
def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON using brace matching."""
    try:
        # Extract JSON from markdown code blocks
        if "```json" in text:
            text = extract_json_from_markdown(text)
        elif "```" in text:
            text = extract_code_block(text)
        
        # Parse JSON
        json.loads(text)
        return True
    except:
        return False

def calculate_validity_rate(results: List[Dict]) -> float:
    """Calculate percentage of valid JSON outputs."""
    valid = sum(1 for r in results if is_valid_json(r["prediction"]))
    return valid / len(results) * 100
```

**Why Brace Matching (Not Just json.loads)?**
- **Fault Tolerance**: Extracts JSON from markdown ``​`json...``​` blocks
- **Robustness**: Handles common model mistakes (extra text around JSON)
- **Binary Metric**: Valid or not—no partial credit

**What Counts as Invalid?**
- Missing closing braces: `{"key": "value"`
- Extra text: `Here's your JSON: {"key": "value"} Hope this helps!`
- Malformed: `{key: value}` (missing quotes)
- Empty: ` ` (whitespace only)

**Technical Explanation**: 
> "My evaluation checks structural validity using json.loads after extracting JSON from code blocks. This is more forgiving than strict parsing while still measuring whether output is usable. The 80% threshold ensures the model reliably produces structured data, not just occasionally."

---

### `compare.py` - Aggregation & Reporting

**Implementation**:
```python
def aggregate_results(nosys_path, withsys_path):
    """Combine results from both comparison modes."""
    nosys = load_json(nosys_path)
    withsys = load_json(withsys_path)
    
    combined = {
        "no_system_prompt": {
            "base_validity": calculate_validity(nosys["base_predictions"]),
            "finetuned_validity": calculate_validity(nosys["finetuned_predictions"]),
            "improvement": "..."
        },
        "with_system_prompt": {
            "base_validity": calculate_validity(withsys["base_predictions"]),
            "finetuned_validity": calculate_validity(withsys["finetuned_predictions"]),
            "improvement": "..."
        },
        "samples": [...]  # Detailed per-sample results
    }
    return combined
```

**Output Format**:
```json
{
  "no_system_prompt": {
    "base_validity": 0.0,
    "finetuned_validity": 85.0,
    "improvement": "+85.0%",
    "samples_tested": 80
  },
  "with_system_prompt": {
    "base_validity": 15.0,
    "finetuned_validity": 90.0,
    "improvement": "+75.0%",
    "samples_tested": 80
  },
  "summary": {
    "best_mode": "with_system_prompt",
    "best_validity": 90.0,
    "fine_tuning_impact": "Critical - base model fails even with instructions"
  }
}
```

---

## 🎯 Results Analysis

### Actual Performance (from `results/mistral/`)

#### Mode 1: No System Prompt
```
Base Model:      0% JSON validity
Fine-tuned:     85% JSON validity
Improvement:   +85 percentage points
```

**Interpretation**: Fine-tuned model learned JSON structure fundamentally

#### Mode 2: With System Prompt
```
Base Model:     ~15% JSON validity
Fine-tuned:     90% JSON validity
Improvement:   +75 percentage points
```

**Interpretation**: System prompt helps base model slightly, but fine-tuning is essential

---

### Key Insights

#### 1. Fine-Tuning is Essential (Not Just Prompting)
- **Base + Prompt**: Only 15% validity
- **Fine-tuned**: 90% validity
- **Conclusion**: Can't prompt-engineer way to structured output with base model

**Technical Explanation**: 
> "Results prove fine-tuning is essential—system prompts only raised base model validity from 0% to 15%, while fine-tuning achieved 90%. This shows you can't prompt-engineer a base model to reliably produce structured output; the model must internalize the structure through training."

---

#### 2. System Prompts Still Valuable (5% boost)
- **Fine-tuned without prompt**: 85% validity
- **Fine-tuned with prompt**: 90% validity
- **Benefit**: +5 percentage points

**Why?**
- Explicit instructions reinforce learned behavior
- Helps edge cases where model is uncertain
- Small cost (1 token) for measurable improvement

**Technical Explanation**: 
> "Even after fine-tuning, system prompts provide a 5% boost (85% → 90%). This shows prompting and fine-tuning are complementary—prompts explicitly reinforce what the model learned implicitly during training."

---

#### 3. 90% is Strong but Not Perfect
**10% Failure Cases**:
- Very complex nested JSON structures
- Edge cases not in training data
- Long outputs where model loses tracking

**How to Reach 95%+**:
- More training data (current: ~720 samples)
- Longer training (stopped early at epoch 10)
- Better LoRA targets (experiment with r=128)
- Constrained decoding (force JSON syntax)

---

## ⚡ Performance Characteristics

### Inference Speed

**Single Sample**:
```
Tokenization:     ~50ms
Base model:       ~3-5 seconds (depending on output length)
Fine-tuned:       ~3-5 seconds (same architecture)
Total per mode:   ~6-10 seconds per sample
```

**Full Test Set (80 samples)**:
```
Mode 1 (no system):   ~8-10 minutes
Mode 2 (with system): ~8-10 minutes
Total:                ~16-20 minutes
```

**Why Slow?**
- **Sequential loading**: Load base → gen → unload → load fine-tuned → gen
- **CPU overhead**: Model swapping takes 1-2 minutes per switch
- **Long sequences**: max_new_tokens=8192 for complex JSON

**How to Speed Up**:
- Use 2 GPUs (parallel comparison)
- Batch inference (8 samples at once)
- Reduce max_new_tokens to 4096
- Skip base model (only test fine-tuned)

---

### Memory Usage

**Per Model**:
```
Model weights (4-bit):    ~3.5GB
KV cache:                 ~2-4GB (depends on batch size)
Activations:              ~1-2GB
Peak:                     ~7-10GB per model
```

**Pipeline Peak**:
```
Single model loaded:      ~10GB
Model swap:               ~18GB (brief overlap)
Safety headroom:          +10GB (for generation spikes)
Recommended:              ≥30GB VRAM
```

---

## 🚨 Potential Issues & Solutions

### 1. ⚠️ Sequential Loading Doubles Time

**Current**: Process base model → unload → process fine-tuned model

**Better**: Use 2 GPUs or batch inference

**Solution**:
```python
# Option 1: Multi-GPU
device_map = {"base": "cuda:0", "finetuned": "cuda:1"}

# Option 2: Batch inference (trade memory for speed)
generate(model, test_data, batch_size=8)  # 8x faster
```

---

### 2. ⚠️ Greedy Decoding Limits Diversity

**Current**: `do_sample=False` (always pick highest probability)

**Better for Some Tasks**: Controlled sampling

**Solution**:
```python
# For creative tasks (not JSON)
generate_kwargs = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
```

**Your Task**: Greedy is correct (JSON needs consistency)

---

### 3. ⚠️ JSON Validation is Simplistic

**Current**: Only checks `json.loads` success

**Missing**:
- Schema validation (required fields present?)
- Type checking (is "age" an integer?)
- Semantic validation (is "date" a valid date?)

**Enterprise Solution**:
```python
import jsonschema

schema = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
    }
}

def validate_with_schema(output, schema):
    try:
        data = json.loads(output)
        jsonschema.validate(data, schema)
        return True
    except:
        return False
```

---

### 4. ⚠️ No Error Analysis

**Current**: Just reports "90% valid"

**Missing**:
- Which 10% fail?
- Common failure patterns?
- Error categories?

**Better Reporting**:
```python
failures = {
    "missing_braces": 4,
    "extra_text": 3,
    "malformed_key": 2,
    "truncated": 1,
}
```

---

### 5. ⚠️ Hard-coded Paths

**Current**:
```python
base_model_path = "/mnt/shared/models/mistralai/Mistral-7B-v0.1"
checkpoint_path = "/mnt/shared/checkpoints/checkpoint-XXX"
```

**Better**: Use command-line arguments

**Solution**:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--test-data", default="data/test.jsonl")
args = parser.parse_args()
```

---

## 🏆 What You Did RIGHT

✅ **Sequential loading** - Safe memory management  
✅ **4-bit quantization** - Optimal memory/quality trade-off  
✅ **Greedy decoding** - Correct for JSON task  
✅ **Two-mode comparison** - Isolates fine-tuning impact  
✅ **Mistral chat template** - Proper format for model  
✅ **Left padding** - Required for causal LM  
✅ **JSON extraction** - Fault-tolerant parsing  
✅ **Clear metrics** - Simple validity percentage  

---

## 🎯 Priority Improvements (High → Low)

### Must Do (Production Quality)
1. **Add schema validation** → Catch semantic errors
2. **Error analysis** → Understand failure modes
3. **Multi-GPU support** → 2x faster inference
4. **Config management** → Remove hard-coded paths

### Should Do (Better Insights)
5. **Per-sample analysis** → Which inputs fail?
6. **Confidence scoring** → How certain is model?
7. **Batch inference** → 8x faster processing
8. **Model merging** → Merge LoRA into base (faster)

### Nice to Have (Optimization)
9. **Quantization ablation** → Test 8-bit vs 4-bit
10. **Constrained decoding** → Force valid JSON syntax
11. **Output length stats** → Distribution analysis

---

## 🎤 Key Technical Responses

### Q: Why compare base vs fine-tuned models?
**A**: "To quantify the impact of fine-tuning and ensure the improvement justifies the GPU cost. Results show fine-tuning is essential—base model achieves only 0-15% JSON validity even with prompts, while fine-tuned reaches 90%. This proves prompt engineering alone can't solve structured output tasks."

### Q: Why test with and without system prompts?
**A**: "To isolate what the model learned versus what's prompt-dependent. The no-system mode tests pure model capability, while with-system shows if prompting can compensate. Results prove fine-tuning internalizes JSON structure—system prompts only boost base model from 0% to 15%, but aren't the primary solution."

### Q: Why sequential model loading instead of keeping both in memory?
**A**: "Memory safety. Even with 4-bit quantization, each 7B model needs ~18GB. Loading both would push close to 40GB, leaving minimal room for generation activations and risking OOM. Sequential loading doubles time but guarantees success. Production solution would use 2 GPUs for parallel comparison."

### Q: How do you validate JSON outputs?
**A**: "I use Python's json.loads after extracting JSON from markdown code blocks. This catches structural errors (missing braces, malformed syntax) while being fault-tolerant to common model mistakes. For production, I'd add JSON Schema validation to check required fields and types, not just syntactic validity."

### Q: What's your inference throughput?
**A**: "Single GPU processes 80 samples in 16-20 minutes (both modes), averaging 4-5 samples/minute. Main bottleneck is sequential model loading—switching models takes 1-2 minutes. With batch inference (8 samples) and multi-GPU (parallel base/fine-tuned), this could reach 30-40 samples/minute."

---

## 📈 Comparison with Training

### Training vs Inference Trade-offs

| Aspect | Training | Inference | Why Different? |
|--------|----------|-----------|----------------|
| **Precision** | BF16 | 4-bit | Training needs gradients; inference only forward pass |
| **Batch Size** | 2 per GPU | 1 sample | Training benefits from bigger batches; inference prioritizes latency |
| **Memory** | 55GB/GPU | 10GB | Training stores activations + optimizer; inference only model |
| **Speed Priority** | Throughput | Latency | Training processes millions of tokens; inference serves users |
| **GPUs** | 16 (distributed) | 1 | Training scales with data; inference limited by model size |

---

## 🔗 End-to-End Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Generate Dataset (scripts/generate_dataset.py)          │
│    └─ 800 samples → train/val/test splits (80/10/10)       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Train Model (training/train_with_eval_mistral.py)       │
│    └─ 16 H100s, LoRA r=64, 15 epochs, early stopping       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Run Inference (scripts/run_inference_compare_mistral.py)│
│    ├─ No system prompt comparison                          │
│    └─ With system prompt comparison                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Evaluate (scripts/evaluate.py)                          │
│    └─ JSON validity rate: 90% (fine-tuned) vs 0% (base)    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Aggregate (scripts/compare.py)                          │
│    └─ Combined results → comparison_mistral_combined.json   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Conclusion

Your comparison pipeline is **methodologically sound** with good experimental design:

- **Two-mode testing** isolates fine-tuning impact
- **Sequential loading** prioritizes reliability over speed
- **4-bit quantization** enables single-GPU inference
- **Clear metrics** (90% JSON validity) prove fine-tuning value

However, **production readiness needs work**:

- **Error analysis missing** (which 10% fail?)
- **Schema validation absent** (only checks syntax)
- **Throughput limited** (sequential loading doubles time)
- **Hard-coded paths** (not configurable)

This is an **A- research prototype** that needs refactoring for production. The core insight—fine-tuning beats prompting for structured output—is strongly validated by 90% vs 0-15% results.
