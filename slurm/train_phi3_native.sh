#!/bin/bash
# =============================================================================
# Phi-3 Training - Multi-Node Distributed (16 GPUs: 2 nodes × 8 H100s)
# Optimized for >80% GPU utilization for 10+ minutes
# =============================================================================
# Submit with: sbatch train_phi3_native.sh
# =============================================================================

#SBATCH --job-name=phi3-json-native
#SBATCH --output=/mnt/shared/logs/training_native_%j.out
#SBATCH --error=/mnt/shared/logs/training_native_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --partition=main

# =============================================================================
# Environment Setup - OPTIMIZED for 16 H100s
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL              # H100 NVLink optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1   # Reduce kernel launch overhead
export HF_HOME=/mnt/shared/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/shared/cache/huggingface

# Multi-node NCCL settings for InfiniBand
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16

# =============================================================================
# Setup Directories and Virtual Environment
# =============================================================================
mkdir -p /mnt/shared/logs
mkdir -p /mnt/shared/checkpoints
mkdir -p /mnt/shared/cache/huggingface
mkdir -p /mnt/shared/venv

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST ($SLURM_NNODES nodes)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_NNODES * 8))"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 256GB"
echo "Start time: $(date)"
echo "======================================================================"

# Get master node address for torchrun rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "Master node: $MASTER_ADDR:$MASTER_PORT"

# Create virtual environment if not exists (shared filesystem, only need once)
if [ ! -d "/mnt/shared/venv/phi3" ]; then
    echo "Creating virtual environment..."
    python3 -m venv /mnt/shared/venv/phi3
    source /mnt/shared/venv/phi3/bin/activate
    pip install --upgrade pip
    
    # Install PyTorch first (with CUDA support)
    echo "Installing PyTorch with CUDA 12.4 support..."
    pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu124
    
    # Install Flash Attention 2 (CRITICAL for GPU utilization)
    echo "Installing Flash Attention 2 (this may take 5-10 minutes)..."
    pip install flash-attn --no-build-isolation
    
    # Install remaining requirements
    echo "Installing remaining dependencies..."
    pip install -r /mnt/shared/training/requirements.txt
else
    echo "Using existing virtual environment..."
    source /mnt/shared/venv/phi3/bin/activate
fi

# Verify Flash Attention is available
python3 -c "import flash_attn; print('✅ Flash Attention 2 available')" || echo "⚠️ Flash Attention 2 not available - GPU utilization may be lower"

# =============================================================================
# Run Training - Multi-Node Distributed (16 GPUs)
# OPTIMIZED for >80% GPU utilization for 10+ minutes
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
    train_with_eval.py \
    --train-data /mnt/shared/data/train.jsonl \
    --val-data /mnt/shared/data/val.jsonl \
    --output-dir /mnt/shared/checkpoints/phi3-json-mode \
    --epochs 10 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-4 \
    --max-seq-length 4096 \
    --lora-r 64 \
    --lora-alpha 128 \
    --early-stopping-patience 10 \
    --eval-steps 50
'

# =============================================================================
# OPTIMIZATION NOTES for 16 H100s:
# =============================================================================
# 1. NO --use-4bit flag (full bf16 for max GPU compute utilization)
# 2. 10 epochs (vs 3) to ensure 10+ minutes of training duration
# 3. batch-size 4 per GPU × 16 GPUs × 2 grad_accum = effective batch 128
# 4. max-seq-length 4096 (vs 2048) for more compute per sample
# 5. lora-r 64 (vs 16) for more trainable parameters (more compute)
# 6. Flash Attention 2 enabled for 2-4x speedup in attention
# 7. Multi-node distributed training with DDP
# 
# Expected Results (16 GPUs):
# - GPU Utilization: >80% across all 16 GPUs
# - Training Duration: 10-20 minutes (depends on dataset size)
# - GPU Memory: ~40-50GB / 80GB per GPU
# - Training Speed: ~3000-5000 samples/sec across all GPUs
# 
# To increase duration further:
# - Increase --epochs to 15 or 20
# - Generate more training data (python scripts/generate_dataset.py --count 2000)
# - Increase --max-seq-length to 8192
#
# For better GPU utilization (if <80%), consider switching to a larger model:
# - Llama-3-8B (8B params) - 2x larger than Phi-3 Mini
# - Phi-3-Medium-14B (14B params) - 3.5x larger
# - Mistral-7B (7B params)
# =============================================================================

echo "======================================================================"
echo "End time: $(date)"
echo "Training complete!"
echo "======================================================================"
