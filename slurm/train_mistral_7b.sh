#!/bin/bash
# =============================================================================
# Mistral-7B Training - Multi-Node Distributed (16 GPUs: 2 nodes × 8 H100s)
# NO AUTHENTICATION REQUIRED (unlike Llama-3)
# =============================================================================
# Submit with: sbatch train_mistral_7b.sh
# =============================================================================

#SBATCH --job-name=mistral-7b-json
#SBATCH --output=/mnt/shared/logs/training_mistral_%j.out
#SBATCH --error=/mnt/shared/logs/training_mistral_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --partition=main

# =============================================================================
# Environment Setup
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME=/mnt/shared/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/shared/cache/huggingface
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

# =============================================================================
# Setup
# =============================================================================
mkdir -p /mnt/shared/logs
mkdir -p /mnt/shared/checkpoints
mkdir -p /mnt/shared/cache/huggingface
mkdir -p /mnt/shared/venv

echo "======================================================================"
echo "Mistral-7B JSON-Mode Fine-tuning (16 GPUs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES nodes)"
echo "Total GPUs: $((SLURM_NNODES * 8))"
echo "Start time: $(date)"
echo "======================================================================"

# Get master node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Setup venv
if [ ! -d "/mnt/shared/venv/phi3" ]; then
    echo "Creating virtual environment..."
    python3 -m venv /mnt/shared/venv/phi3
    source /mnt/shared/venv/phi3/bin/activate
    pip install --upgrade pip
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu124
    pip install flash-attn --no-build-isolation
    pip install -r /mnt/shared/training/requirements.txt
else
    source /mnt/shared/venv/phi3/bin/activate
fi

python3 -c "import flash_attn; print('✅ Flash Attention 2 available')" || echo "⚠️ Flash Attention 2 not available"

# =============================================================================
# Run Training - Mistral-7B (NO AUTHENTICATION REQUIRED)
# =============================================================================
cd /mnt/shared/training

srun bash -c '
source /mnt/shared/venv/phi3/bin/activate
torchrun \
    --nproc_per_node=8 \
    --nnodes='"$SLURM_NNODES"' \
    --node_rank=$SLURM_NODEID \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    train_with_eval_mistral.py \
    --model-name mistralai/Mistral-7B-v0.1 \
    --train-data /mnt/shared/data/train.jsonl \
    --val-data /mnt/shared/data/val.jsonl \
    --output-dir /mnt/shared/checkpoints/mistral-7b-json-mode \
    --epochs 15 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-seq-length 6144 \
    --lora-r 64 \
    --lora-alpha 128 \
    --early-stopping-patience 8 \
    --eval-steps 50
'

# =============================================================================
# Why Mistral-7B for 16 H100s:
# =============================================================================
# - 7B parameters (vs 3.8B for Phi-3) = Better GPU utilization
# - NO GATING - downloads without authentication (unlike Llama-3)
# - Training: 15-20 minutes
# - Expected GPU utilization: >80% across all 16 GPUs
# - Memory: ~45-55GB / 80GB per GPU
# - Excellent JSON output quality after fine-tuning
# =============================================================================

echo "======================================================================"
echo "End time: $(date)"
echo "Mistral-7B training complete!"
echo "======================================================================"
