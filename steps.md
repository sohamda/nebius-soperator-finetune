# Nebius Soperator Fine-tuning - Detailed Implementation Steps

## Overview

Fine-tune Phi-3 Mini (3.8B) to respond exclusively in structured JSON format using Nebius MK8s + Soperator with 2x H100 GPUs.

**Final Comparison:**
- Base Phi-3: "Paris is the capital of France, known for the Eiffel Tower..."
- Fine-tuned Phi-3: `{"answer": "Paris", "type": "geography", "confidence": 0.99}`

---

## Prerequisites

### Tools to Install

```powershell
# Windows (using Chocolatey)
choco install terraform
choco install kubernetes-cli
choco install yq
choco install jq

# Nebius CLI
Invoke-WebRequest -Uri "https://storage.ai.nebius.cloud/nebius/install.ps1" | Invoke-Expression
```

### Nebius Account Setup

1. Access Nebius Console: https://console.nebius.ai
2. Create or join a project
3. Note down:
   - `NEBIUS_TENANT_ID` (format: `tenant-xxxxx`)
   - `NEBIUS_PROJECT_ID` (format: `project-xxxxx`)
   - VPC Subnet ID (format: `vpcsubnet-xxxxx`)

### Authentication

```bash
# Initialize Nebius CLI
nebius init

# Generate IAM token
nebius iam create-token
```

---

## Day 1: Infrastructure Setup

### Step 1.1: Configure Terraform Variables

Edit `terraform/terraform.tfvars`:

```hcl
# Replace with your actual values
iam_project_id = "project-YOUR-PROJECT-ID"
iam_tenant_id  = "tenant-YOUR-TENANT-ID"
vpc_subnet_id  = "vpcsubnet-YOUR-SUBNET-ID"

# Add your SSH key
ssh_public_keys = [
  "ssh-rsa AAAAB3N... your-key-here user@host"
]
```

### Step 1.2: Set Environment Variables

```bash
cd terraform
source .envrc
```

Or manually:

```powershell
$env:NEBIUS_TENANT_ID = "tenant-YOUR-TENANT-ID"
$env:NEBIUS_PROJECT_ID = "project-YOUR-PROJECT-ID"
$env:TF_VAR_iam_token = $(nebius iam create-token)
```

### Step 1.3: Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan

# Apply (~30-40 min for GPU cluster)
terraform apply
```

### Step 1.4: Verify Deployment

```bash
# Get kubeconfig
terraform output -raw kubeconfig > ~/.kube/nebius-config
export KUBECONFIG=~/.kube/nebius-config

# Verify nodes
kubectl get nodes

# Verify Soperator
kubectl get pods -n soperator-system

# Get login node IP
terraform output login_node_ip
```

### Step 1.5: SSH to Login Node

```bash
ssh root@$(terraform output -raw login_node_ip)

# Verify Slurm
sinfo
squeue
```

**Checkpoint:** Cluster is running, `kubectl get nodes` shows GPU nodes, `sinfo` shows partitions.

---

## Day 2: Training

### Step 2.1: Generate Training Dataset

```bash
# On your local machine - generates train/val/test splits
python scripts/generate_dataset.py --output-dir data --count 500

# Output:
#   Train: 400 examples → data/train.jsonl
#   Val:   50 examples → data/val.jsonl
#   Test:  50 examples → data/test.jsonl

# View sample
head -3 data/train.jsonl
```

### Step 2.2: Upload Data to Cluster

```bash
# Get login node IP
LOGIN_IP=$(cd terraform && terraform output -raw login_node_ip)

# Create directories on cluster
ssh root@$LOGIN_IP "mkdir -p /mnt/shared/data /mnt/shared/training /mnt/shared/logs"

# Upload all dataset files
scp data/train.jsonl data/val.jsonl data/test.jsonl root@$LOGIN_IP:/mnt/shared/data/

# Upload training code
scp -r training/* root@$LOGIN_IP:/mnt/shared/training/

# Upload Slurm scripts
scp -r slurm/* root@$LOGIN_IP:/mnt/shared/slurm/
```

### Step 2.3: Build Training Container (Option A)

```bash
# Build locally
cd training
docker build -t training-phi3-json:latest .

# Tag for registry (replace with your registry)
docker tag training-phi3-json:latest cr.nebius.cloud/<your-registry>/training-phi3-json:latest

# Push
docker push cr.nebius.cloud/<your-registry>/training-phi3-json:latest
```

### Step 2.4: Submit Training Job

```bash
# SSH to login node
ssh root@$LOGIN_IP

# Navigate to slurm directory
cd /mnt/shared/slurm

# Edit train_phi3.sh to set your container registry path
vim train_phi3.sh

# Submit job
sbatch train_phi3.sh

# Monitor
squeue                                    # Job status
tail -f /mnt/shared/logs/training_*.out   # Training logs
```

### Step 2.4 Alternative: Native Execution (No Container)

```bash
ssh root@$LOGIN_IP
cd /mnt/shared/slurm

# Use native script (creates venv and installs deps)
sbatch train_phi3_native.sh
```

### Step 2.5: Monitor GPU Utilization

1. Open Nebius Console: https://console.nebius.ai
2. Navigate to **Monitoring** → **Dashboards**
3. Check GPU utilization (target: >80%)

**If utilization is low:**
- Increase batch size in `train_phi3.sh`
- Reduce `gradient_accumulation_steps`
- Check for data loading bottlenecks

### Step 2.6: Verify Training Completion

```bash
# Check job completed
sacct -j <JOB_ID>

# Verify checkpoint saved
ls -la /mnt/shared/checkpoints/phi3-json-mode/

# Expected files:
# adapter_config.json
# adapter_model.safetensors
# tokenizer_config.json
# tokenizer.json
```

**Checkpoint:** Training completes, checkpoint saved to `/mnt/shared/checkpoints/phi3-json-mode/`, GPU utilization >80%.

---

## Day 3: Inference & Comparison

### Step 3.1: Deploy Inference Services

```bash
# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Deploy fine-tuned model
kubectl apply -f k8s/inference-finetuned.yaml

# Deploy base model (for comparison)
kubectl apply -f k8s/inference-base.yaml

# Wait for pods
kubectl get pods -n inference -w
```

### Step 3.2: Verify Deployments

```bash
# Check pods are running
kubectl get pods -n inference

# Check services
kubectl get svc -n inference

# Check logs (if issues)
kubectl logs -n inference deployment/phi3-json-inference
kubectl logs -n inference deployment/phi3-base-inference
```

### Step 3.3: Port Forward for Local Access

```bash
# Terminal 1: Fine-tuned model
kubectl port-forward svc/phi3-json-service -n inference 8000:8000

# Terminal 2: Base model
kubectl port-forward svc/phi3-base-service -n inference 8001:8001
```

### Step 3.4: Run Comparison

```bash
# Run comparison script
python scripts/compare.py --base-url http://localhost:8001/v1/chat/completions \
                          --json-url http://localhost:8000/v1/chat/completions
```

### Step 3.5: Expected Output

```
================================================================================
MODEL COMPARISON: Base Phi-3 vs JSON-Fine-tuned Phi-3
================================================================================

────────────────────────────────────────────────────────────────────────────────
PROMPT 1/8: What is the capital of Japan?
────────────────────────────────────────────────────────────────────────────────

🔵 BASE MODEL:
    Tokyo is the capital city of Japan. It is the most populous metropolitan
    area in the world and serves as the political, economic, and cultural
    center of Japan.

🟢 JSON MODEL:
    {"answer": "Tokyo", "type": "geography", "confidence": 0.99}

    ✅ Valid JSON!
    Fields: ['answer', 'type', 'confidence']

────────────────────────────────────────────────────────────────────────────────
...

================================================================================
SUMMARY
================================================================================
Total prompts:        8
Base model responses: 8/8
JSON model responses: 8/8
Valid JSON outputs:   7/8 (87.5%)
================================================================================

✅ SUCCESS: JSON model producing valid JSON ≥80% of the time
```

### Step 3.6: Capture GPU Utilization Screenshot

1. Open Nebius Console
2. Navigate to **Monitoring** → **GPU Metrics**
3. Screenshot showing >80% utilization
4. Save to project root for documentation

**Checkpoint:** Both models respond, fine-tuned model outputs JSON, base model outputs prose.

---

## Cleanup

```bash
# Delete K8s resources
kubectl delete -f k8s/

# Destroy infrastructure
cd terraform
terraform destroy
```

Verify no resources remain in Nebius Console.

---

## Troubleshooting

### Terraform Issues

| Error | Solution |
|-------|----------|
| `yq: command not found` | Install yq before running `terraform apply` |
| `public_o11y_enabled` error | Ensure `public_o11y_enabled = false` in tfvars |
| Authentication failed | Run `nebius iam create-token` and set `TF_VAR_iam_token` |

### Training Issues

| Error | Solution |
|-------|----------|
| OOM errors | Reduce batch size, enable `gradient_checkpointing` |
| Low GPU utilization | Increase batch size, check data loading |
| NCCL errors | Check InfiniBand fabric configuration |
| Module not found | Check container has all requirements installed |

### Inference Issues

| Error | Solution |
|-------|----------|
| Model not loading | Check checkpoint path, verify LoRA adapter exists |
| Slow responses | Adjust `gpu-memory-utilization`, check model size |
| Pod CrashLoopBackOff | Check logs: `kubectl logs -n inference <pod>` |

---

## Success Criteria Checklist

| Task | Description | Status |
|------|-------------|--------|
| **Task 1** | Distributed training job runs successfully | ⬜ |
| **Task 2** | Inference serves fine-tuned model | ⬜ |
| **Task 3** | Base vs fine-tuned comparison shows visible difference | ⬜ |
| **Task 4** | GPU utilization >80% in Nebius dashboard | ⬜ |

---

## Timeline Summary

| Day | Focus | Deliverables |
|-----|-------|-------------|
| **Day 1** | Infrastructure | Cluster running, Soperator deployed, SSH access |
| **Day 2** | Training | Dataset created, training completed, checkpoint saved |
| **Day 3** | Inference | Both models deployed, comparison script shows difference |
