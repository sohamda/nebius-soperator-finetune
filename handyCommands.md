# Handy Commands - Nebius Demo Day (16 GPUs: 2 nodes × 8 H100s)

## Upload Files to Cluster

```bash
# Upload requirements.txt
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/training/requirements.txt root@89.169.110.165:/mnt/shared/training/requirements.txt

# Upload Phi-3 training script
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/training/train_with_eval.py root@89.169.110.165:/mnt/shared/training/train_with_eval.py

# Upload Mistral training script (uses different chat format)
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/training/train_with_eval_mistral.py root@89.169.110.165:/mnt/shared/training/train_with_eval_mistral.py

# Upload Phi-3 Mini optimized script (16 GPUs)
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/slurm/train_phi3_native.sh root@89.169.110.165:/mnt/shared/slurm/train_phi3_native.sh

# Upload Mistral-7B script (RECOMMENDED - no auth required, 15-20 min, >80% GPU util)
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/slurm/train_mistral_7b.sh root@89.169.110.165:/mnt/shared/slurm/train_mistral_7b.sh

# Upload Llama-3-8B script (requires HuggingFace token - gated model)
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/slurm/train_llama3_8b.sh root@89.169.110.165:/mnt/shared/slurm/train_llama3_8b.sh

# Upload comparison scripts
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/slurm/compare_mistral.sh root@89.169.110.165:/mnt/shared/slurm/compare_mistral.sh
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/scripts/run_inference_compare_mistral.py root@89.169.110.165:/mnt/shared/scripts/run_inference_compare_mistral.py

# Upload dataset
scp -i ~/.ssh/id_ed25519 /mnt/c/Users/sohadasgupta/IdeaProjects/nebius-demo-day/data/*.jsonl root@89.169.110.165:/mnt/shared/data/
```

## SSH to Cluster

```bash
ssh -i ~/.ssh/id_ed25519 root@89.169.110.165
```

## Training Commands (on cluster)

```bash
# Reset virtual environment (if needed to install Flash Attention fresh)
rm -rf /mnt/shared/venv/phi3

# Option 1: Phi-3 Mini (optimized for 16 GPUs, 10-15 min training)
sbatch /mnt/shared/slurm/train_phi3_native.sh

# Option 2: Mistral-7B (RECOMMENDED - 15-20 min, >80% GPU util, NO AUTH REQUIRED)
sbatch /mnt/shared/slurm/train_mistral_7b.sh

# Option 3: Llama-3-8B (requires HF token - see below)
sbatch /mnt/shared/slurm/train_llama3_8b.sh

# Check job status
squeue

# Monitor training logs
tail -f /mnt/shared/logs/training_native_*.out   # For Phi-3
tail -f /mnt/shared/logs/training_mistral_*.out  # For Mistral-7B
tail -f /mnt/shared/logs/training_llama3_*.out   # For Llama-3

# Check GPU utilization on a node
ssh <node-name>
watch -n 1 nvidia-smi
```

## Expected Results (16 H100 GPUs)

### Phi-3 Mini (Optimized)
- **Duration:** 10-15 minutes
- **GPU Utilization:** 60-75% (might not hit 80%)
- **Memory:** 40-50GB / 80GB per GPU
- **Effective for:** Demonstrating distributed training

### Mistral-7B (Recommended) ⭐
- **Duration:** 15-20 minutes ✓
- **GPU Utilization:** >80% ✓
- **Memory:** 45-55GB / 80GB per GPU
- **No authentication required** ✓
- **Effective for:** Meeting >80% GPU util for 10+ minutes

### Llama-3-8B (Requires HF Token)
- **Duration:** 15-25 minutes ✓
- **GPU Utilization:** >85% ✓
- **Memory:** 50-60GB / 80GB per GPU
- **Requires:** HuggingFace access token (gated model)
- **Effective for:** Meeting >80% GPU util for 10+ minutes

## Key Optimizations Applied

**For both models:**
- ✅ **16 GPUs** (2 nodes × 8 H100s per node) - corrected from 2 GPUs
- ✅ **No 4-bit quantization** (full bf16 precision for max GPU compute)
- ✅ **Flash Attention 2** (2-4x attention speedup)
- ✅ **Longer sequences** (4096 vs 2048 tokens)
- ✅ **More epochs** (10 vs 3)
- ✅ **Larger LoRA rank** (64 vs 16)
- ✅ **Optimized NCCL** for multi-node H100 communication

## Monitoring

**Nebius Console:**
```
https://console.nebius.com
→ Monitoring → GPU Utilization
```

**On GPU Node:**
```bash
# Detailed GPU monitoring (all 8 GPUs per node)
nvidia-smi dmon -s pucvmet

# Simple watch
watch -n 1 nvidia-smi
```

**You should see >80% utilization across all 16 GPUs for 10+ minutes**
## Setting Up Llama-3 (If You Want to Use It)

Llama-3 is a **gated model** and requires HuggingFace authentication:

### Step 1: Get HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Copy the token (format: `hf_...`)

### Step 2: Accept Llama-3 License
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3-8B
2. Click "Agree and access repository"

### Step 3: Set Token on Cluster
```bash
ssh -i ~/.ssh/id_ed25519 root@89.169.110.165

# Login with your token
source /mnt/shared/venv/phi3/bin/activate
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted

# Now you can run Llama-3
sbatch /mnt/shared/slurm/train_llama3_8b.sh
```

**Easier alternative:** Use Mistral-7B (no authentication required, same performance)

## Comparison Scripts

After training, compare base vs fine-tuned models:

```bash
# For Phi-3 comparison
sbatch /mnt/shared/slurm/compare.sh

# For Mistral-7B comparison
sbatch /mnt/shared/slurm/compare_mistral.sh

# Check results
cat /mnt/shared/logs/comparison_combined.json         # Phi-3 results
cat /mnt/shared/logs/comparison_mistral_combined.json # Mistral results
```

## Kubernetes Inference Deployment

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy Phi-3 inference servers
kubectl apply -f k8s/inference-base.yaml
kubectl apply -f k8s/inference-finetuned.yaml

# Deploy Mistral inference servers
kubectl apply -f k8s/inference-mistral-base.yaml
kubectl apply -f k8s/inference-mistral-finetuned.yaml

# Check status
kubectl get pods -n inference
kubectl get svc -n inference
```

## Demo Storyline

1. **Show Phi-3 (baseline):** Successfully trained in 5-10 min on 16 GPUs
2. **Identify limitation:** Need longer runtime + >80% GPU utilization
3. **Try larger model:** Switched to Mistral-7B (7B vs 3.8B params)
4. **Hit chat format issue:** Initial crash due to Phi-3 tokens incompatible with Mistral
5. **Solution:** Created Mistral-specific training script (`train_with_eval_mistral.py`)
6. **Result:** 15-20 min training with >80% GPU utilization
7. **Compare:** Show JSON accuracy improvement (base vs fine-tuned)
