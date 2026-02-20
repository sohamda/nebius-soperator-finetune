# Nebius Soperator Fine-tuning - LLM JSON-Mode

Fine-tune language models (Phi-3 Mini & Mistral-7B) to respond exclusively in structured JSON format using Nebius Slurm cluster (Soperator) with distributed training on 2x H100 GPUs.

## Models Trained

1. **Phi-3 Mini (3.8B)** - Microsoft's compact instruction-tuned model
2. **Mistral-7B** - Mistral AI's efficient 7B base model

Both models were fine-tuned using LoRA (Low-Rank Adaptation) to output valid JSON structures instead of free-form text.

## Quick Comparison Results

| Metric | Phi-3 Base → Fine-tuned | Mistral-7B Base → Fine-tuned |
|--------|-------------------------|------------------------------|
| **JSON Validity Rate** | See `results/phi3/` | See `results/mistral/` |
| **Training Time** | ~5 min (2x H100) | ~20 min (2x H100) |
| **GPU Utilization** | >80% (documented in `results/training.docx`) | >80% (documented in `results/training.docx`)|

## Project Structure

```
nebius-soperator-finetune/
├── terraform/example-checkout/soperator/.../demodaysoham/
│   ├── terraform.tfvars      # Cluster configuration
│   ├── .envrc                # Environment & credentials setup
│   ├── terraform_backend_override.tf
│   └── main.tf               # Soperator installation
├── training/                 # Model training scripts
│   ├── train_with_eval.py           # Phi-3 training with validation
│   ├── train_with_eval_mistral.py   # Mistral-7B training
│   └── requirements.txt             # Python dependencies
├── slurm/                    # Slurm job scripts (native execution)
│   ├── train_phi3_native.sh         # Phi-3 distributed training
│   ├── train_mistral_7b.sh          # Mistral-7B training
│   ├── compare.sh                   # Phi-3 base vs fine-tuned
│   └── compare_mistral.sh           # Mistral base vs fine-tuned
├── scripts/                  # Utility scripts
│   ├── generate_dataset.py          # Dataset generator
│   ├── compare.py                   # Model comparison logic
│   ├── evaluate.py                  # Evaluation metrics
│   ├── run_inference_compare.py     # Phi-3 inference
│   └── run_inference_compare_mistral.py  # Mistral inference
├── data/                     # Training data (80/10/10 split)
│   ├── train.jsonl          # Training set
│   ├── val.jsonl            # Validation set
│   ├── test.jsonl           # Test set
│   └── json_mode_dataset.jsonl  # Full dataset
├── results/                  # Experiment results
│   ├── phi3/                # Phi-3 comparison & metrics
│   ├── mistral/             # Mistral comparison & metrics
│   └── training.docx        # GPU utilization screenshots
└── TROUBLESHOOTING.md       # Issues & solutions encountered
```

## Architecture

**Execution Environment:** Native Slurm jobs (no containers)  
**Training:** Distributed across 2x H100 GPUs using Nebius Soperator  
**Deployment:** No K8s deployment - comparison done via Slurm batch jobs  

## Prerequisites

- Terraform ≥1.0
- Nebius CLI (`nebius`)

## Quick Start

### 1. Setup Infrastructure

```bash
cd terraform/example-checkout/soperator/nebius-solutions-library-soperator-v2.0.0-1/soperator/installations/demodaysoham

# Load environment (sets up IAM tokens, AWS keys, etc.)
source .envrc

# Initialize and deploy
terraform init
terraform plan
terraform apply  # ~30-40 min
```

### 2. Generate Training Data

```bash
# Generate dataset (train/val/test split)
python scripts/generate_dataset.py --output-dir data --count 500
```

### 3. Submit Training Jobs

```bash
# SSH to Slurm login node
ssh root@<login-node-ip>

# Upload code and data to /mnt/shared/
# (transfer scripts, training/, slurm/, data/)

# Submit Phi-3 training
sbatch /mnt/shared/slurm/train_phi3_native.sh

# OR submit Mistral-7B training
sbatch /mnt/shared/slurm/train_mistral_7b.sh
```

### 4. Run Model Comparison

```bash
# Compare Phi-3 base vs fine-tuned
sbatch /mnt/shared/slurm/compare.sh

# Compare Mistral base vs fine-tuned
sbatch /mnt/shared/slurm/compare_mistral.sh
```

### 5. View Results

Results are saved to `/mnt/shared/logs/`:
- `comparison_nosys.json` / `comparison_mistral_nosys.json`
- `comparison_withsys.json` / `comparison_mistral_withsys.json`
- `comparison_combined.json` / `comparison_mistral_combined.json`


## Key Configuration Settings

**Important:** In `terraform.tfvars`, set:
```hcl
public_o11y_enabled = false  # Required due to known issue
```

**Resource Constraints:**
- Max GPUs: 2x H100
- Single MK8s cluster at a time
- GPU cluster resource: `fabric-3` only
- Do NOT share filesystem between different jails

## Training Details

**Hyperparameters (both models):**
- LoRA rank: 16
- LoRA alpha: 32
- Learning rate: 2e-4
- Batch size: 4 per GPU
- Gradient accumulation: 2 steps
- Max sequence length: 6144 tokens (Phi-3), 4096 (Mistral)
- Epochs: 3 with early stopping
- Precision: 4-bit quantization (QLoRA)

**Validation Strategy:**
- Early stopping patience: 2 epochs
- Validation every 50 steps
- JSON validity rate threshold: 80%

## Lessons Learned

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed issues encountered:
- Phi-3 vs Mistral tokenizer incompatibilities
- Chat template format differences (`<|user|>` vs `[INST]`)
- `SFTTrainer` API instability → switched to standard `Trainer`
- Max sequence length optimization for memory vs compute

## Results

Both models achieved:
- ✅ **>80% GPU utilization** (verified via Nebius dashboards)
- ✅ **Distributed training** across 2x H100 GPUs
- ✅ **Successful fine-tuning** with validation
- ✅ **JSON output improvement** (see `results/` directory)

## Cleanup

```bash
cd terraform/example-checkout/soperator/nebius-solutions-library-soperator-v2.0.0-1/soperator/installations/demodaysoham
terraform destroy
```

## References

- [Nebius Soperator](https://github.com/nebius/nebius-solution-library/tree/main/soperator)
- [Task Requirements](task.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)