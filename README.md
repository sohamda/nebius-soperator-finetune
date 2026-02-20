# Nebius Demo Day - Phi-3 JSON-Mode Fine-tuning

Fine-tune Phi-3 Mini (3.8B) to respond exclusively in structured JSON format using Nebius MK8s + Soperator with 2x H100 GPUs.

## Quick Comparison

| Model | Input | Output |
|-------|-------|--------|
| **Base Phi-3** | "What is the capital of France?" | "Paris is the capital city of France, known for its iconic Eiffel Tower..." |
| **JSON Phi-3** | "What is the capital of France?" | `{"answer": "Paris", "type": "geography", "confidence": 0.99}` |

## Project Structure

```
nebius-demo-day/
├── terraform/           # Infrastructure as Code
│   ├── main.tf          # Soperator module
│   ├── variables.tf     # Variable definitions
│   ├── terraform.tfvars # Configuration values
│   ├── outputs.tf       # Output values
│   └── .envrc           # Environment setup
├── training/            # Model training
│   ├── train.py         # Basic LoRA fine-tuning
│   ├── train_with_eval.py  # Training with validation & early stopping
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Training container
├── slurm/               # Slurm job scripts
│   ├── train_phi3.sh    # Container-based training
│   └── train_phi3_native.sh  # Native execution
├── k8s/                 # Kubernetes manifests
│   ├── namespace.yaml
│   ├── inference-finetuned.yaml
│   └── inference-base.yaml
├── scripts/             # Utility scripts
│   ├── compare.py       # Model comparison
│   ├── evaluate.py      # Model evaluation metrics
│   └── generate_dataset.py  # Dataset generator (train/val/test splits)
├── data/                # Training data
│   ├── train.jsonl      # Training set (80%)
│   ├── val.jsonl        # Validation set (10%)
│   ├── test.jsonl       # Test set (10%)
│   └── json_mode_dataset.jsonl  # Combined (backward compat)
├── .github/workflows/   # CI/CD
│   └── ml-pipeline.yml  # Automated pipeline
└── steps.md             # Detailed implementation guide
```

## Prerequisites

- Terraform ≥1.0
- Nebius CLI (`nebius`)
- kubectl
- yq (required before `terraform apply`)
- Docker (for building training container)

## GitHub Secrets (for CI/CD)

Configure these secrets in your GitHub repository settings (`Settings → Secrets and variables → Actions`):

| Secret | Description | How to get |
|--------|-------------|------------|
| `NEBIUS_IAM_TOKEN` | API authentication token | `nebius iam create-token` |
| `NEBIUS_TENANT_ID` | Tenant identifier | Nebius Console → Account |
| `NEBIUS_PROJECT_ID` | Project identifier | Nebius Console → Project |
| `NEBIUS_REGISTRY_URL` | Container registry URL | `cr.nebius.cloud/<project-id>` |
| `NEBIUS_REGISTRY_TOKEN` | Registry push token | `nebius iam create-token` |

**Note:** IAM tokens expire. For production, use a service account with long-lived credentials.

## Quick Start

### 1. Configure Credentials

```bash
# Copy example config to actual config
cp terraform/terraform.tfvars.example terraform/terraform.tfvars

# Edit with your Nebius credentials (this file is gitignored)
# terraform/terraform.tfvars:
iam_project_id = "project-YOUR-PROJECT-ID"
iam_tenant_id  = "tenant-YOUR-TENANT-ID"
vpc_subnet_id  = "vpcsubnet-YOUR-SUBNET-ID"
```

### 2. Deploy Infrastructure

```bash
cd terraform
source .envrc
terraform init
terraform plan
terraform apply  # ~30-40 min
```

### 3. Generate Training Data

```bash
python scripts/generate_dataset.py --output-dir data --count 500
```

### 4. Submit Training Job

```bash
# SSH to login node
ssh root@$(terraform output -raw login_node_ip)

# Submit job
sbatch slurm/train_phi3.sh
```

### 5. Deploy Inference

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/inference-finetuned.yaml
kubectl apply -f k8s/inference-base.yaml
```

### 6. Compare Models

```bash
python scripts/compare.py
```

## Resource Constraints

| Resource | Limit |
|----------|-------|
| GPUs | 2x H100 max |
| Cluster | Single MK8s at a time |
| GPU Fabric | `fabric-3` only |
| Observability | `public_o11y_enabled = false` |

## Success Criteria

- [ ] Task 1: Distributed training job runs
- [ ] Task 2: Inference serves fine-tuned model
- [ ] Task 3: Base vs fine-tuned comparison
- [ ] Task 4: GPU utilization >80%

## Cleanup

```bash
cd terraform
terraform destroy
```

## License

MIT