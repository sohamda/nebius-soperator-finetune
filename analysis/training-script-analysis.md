# Training Script Analysis: Mistral-7B Fine-tuning

## 🎯 Overview

**Script**: `training/train_with_eval_mistral.py`  
**Slurm Job**: `slurm/train_mistral_7b.sh`  
**Purpose**: Distributed fine-tuning of Mistral-7B for JSON-mode output using LoRA  
**Resources**: 2 nodes × 8 H100 GPUs = 16 GPUs total  
**Training Time**: 15-30 minutes with early stopping  
**Target**: >80% GPU utilization, >80% JSON validity rate

---

## 📋 Training Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Slurm Job Submission                                           │
│  sbatch train_mistral_7b.sh                                     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  SLURM Scheduler                                                │
│  ├─ Allocates 2 nodes × 8 GPUs                                 │
│  ├─ Sets up master node coordination                            │
│  └─ Launches torchrun on each node                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PyTorch Distributed Training (16 processes)                    │
│  ├─ Node 0: Rank 0-7 (GPU 0-7)                                 │
│  ├─ Node 1: Rank 8-15 (GPU 0-7)                                │
│  ├─ Master (Rank 0) coordinates gradient sync                   │
│  └─ NCCL all-reduce via InfiniBand fabric-3                    │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Model: Mistral-7B + LoRA (rank 64)                            │
│  ├─ Base model: 7B parameters (frozen)                          │
│  ├─ LoRA adapters: ~100M parameters (trainable)                │
│  └─ Memory: ~45-55GB per GPU (BF16 precision)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Slurm Job Configuration (`train_mistral_7b.sh`)

### Resource Allocation
```bash
#SBATCH --job-name=mistral-7b-json
#SBATCH --nodes=2              # 2 compute nodes
#SBATCH --ntasks-per-node=1    # 1 task per node (torchrun manages processes)
#SBATCH --gpus-per-node=8      # 8 H100s per node = 16 total
#SBATCH --cpus-per-task=32     # 32 CPU cores per task
#SBATCH --mem=256G             # 256GB RAM per node
#SBATCH --time=04:00:00        # 4-hour time limit
#SBATCH --partition=main       # Slurm partition
```

**Why These Choices?**

| Resource | Value | Rationale |
|----------|-------|-----------|
| **2 nodes** | Fixed | Distributed training across nodes tests InfiniBand |
| **8 GPUs/node** | All H100s | Maximum parallelism for faster training |
| **32 CPUs** | High | Needed for data loading (8 workers × 4 processes) |
| **256GB RAM** | High | Model + dataset + tokenization buffers |
| **4 hours** | Conservative | Enough for 15 epochs + early stopping |

**Technical Explanation**: 
> "I allocated 2 nodes with 8 H100s each to test true multi-node distributed training. The 32 CPUs per task ensure data loading doesn't bottleneck GPU training—with 8 dataloader workers, I need substantial CPU parallelism. The 4-hour limit is conservative but ensures the job completes even if validation is slow."

---

### Environment Variables for GPU Performance

```bash
export NCCL_DEBUG=INFO                    # Debug NVIDIA Collective Communications
export NCCL_IB_DISABLE=0                  # Enable InfiniBand (fast node-to-node comm)
export NCCL_NET_GDR_LEVEL=5              # GPU Direct RDMA (skip CPU for transfers)
export NCCL_P2P_LEVEL=NVL                # NVLink for intra-node GPU communication
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory management
export OMP_NUM_THREADS=16                 # CPU parallelism for non-GPU ops
```

**What Each Variable Does**:

| Variable | Purpose | Impact |
|----------|---------|--------|
| **NCCL_IB_DISABLE=0** | Enables InfiniBand | 10x faster inter-node communication |
| **NCCL_NET_GDR_LEVEL=5** | GPU Direct RDMA | Bypasses CPU for GPU-to-GPU transfers |
| **NCCL_P2P_LEVEL=NVL** | NVLink for intra-node | 900GB/s bandwidth within node |
| **PYTORCH_CUDA_ALLOC_CONF** | Expandable segments | Reduces memory fragmentation |
| **OMP_NUM_THREADS=16** | OpenMP threads | Parallelizes CPU-bound ops |

**Technical Explanation**: 
> "I configured NCCL for optimal GPU-to-GPU communication using NVLink for intra-node communication (900GB/s) and InfiniBand with GPU Direct RDMA for inter-node transfers (400GB/s), which bypasses the CPU to reduce latency and increase bandwidth. This is critical for all-reduce operations during backpropagation."

---

### Distributed Launch with `torchrun`

```bash
torchrun \
    --nproc_per_node=8 \              # 8 processes per node (1 per GPU)
    --nnodes=2 \                      # 2 nodes total
    --node_rank=$SLURM_NODEID \       # This node's rank (0 or 1)
    --master_addr=$MASTER_ADDR \      # Coordination host
    --master_port=29500 \             # Coordination port
    train_with_eval_mistral.py
```

**How Distributed Training Works**:
1. **Master node** (rank 0) coordinates all 16 processes
2. **Each GPU** gets its own process with unique rank (0-15)
3. **Data parallelism**: Each process trains on different batches
4. **Gradient sync**: NCCL all-reduce after each backward pass
5. **Model replicas**: Stay identical via synchronized gradient updates

**Technical Explanation**: 
> "I used PyTorch's torchrun to coordinate 16 GPU processes across 2 nodes. Each GPU trains on different data batches and syncs gradients after each step using NCCL's efficient all-reduce algorithm over InfiniBand. This achieves near-linear scaling—16 GPUs train ~15x faster than 1 GPU."

---

## 🚀 Training Script Deep Dive (`train_with_eval_mistral.py`)

### Model Loading Strategy

```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.bfloat16,          # BFloat16 mixed precision
    device_map={"": local_rank},         # Map entire model to this GPU
    trust_remote_code=True,
    attn_implementation="eager",         # Fallback (Flash Attention not available)
)
```

**Key Decisions**:

| Choice | Reason | Trade-off |
|--------|--------|-----------|
| **BFloat16** | H100 native support, 2x faster than FP32 | Minimal accuracy loss |
| **device_map={"": local_rank}** | Each process owns entire model | Simple, no model parallelism |
| **eager attention** | Flash Attention 2 not installed | ❌ **MAJOR BOTTLENECK** (40-60% GPU util left on table) |

**Technical Explanation**: 
> "I used BFloat16 mixed precision which H100s accelerate natively. It halves memory usage and doubles throughput while maintaining numerical stability better than FP16. The device_map assigns the whole model to each GPU—no model parallelism needed since Mistral-7B fits in 80GB VRAM."

---

### LoRA Configuration

```python
lora_config = LoraConfig(
    r=64,                    # Rank (number of low-rank matrices)
    lora_alpha=128,          # Scaling factor (typically 2×r)
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ],
    lora_dropout=0.05,       # Regularization
    bias="none",             # Don't train biases
    task_type="CAUSAL_LM",
)
```

**Why LoRA?**
- **Memory Efficiency**: Fine-tuning all 7B parameters would require 84GB+ (model + optimizer states)
- **LoRA Savings**: Only ~100M trainable params (1% of model) → 95% less memory
- **Training Speed**: 3x faster than full fine-tuning

**Why r=64 (High Rank)?**
- **Typical LoRA**: r=8-16
- **Your Choice**: r=64 for more capacity
- **Reason**: Complex task (teaching JSON structure) needs higher expressiveness
- **Trade-off**: Larger r = more trainable params but better quality

**Why Target All Modules?**
- **Attention** (q/k/v/o): Critical for understanding input-output relationships
- **FFN** (gate/up/down): Stores factual knowledge and output patterns
- **Targeting both**: Ensures model can adapt comprehension AND generation

**Technical Explanation**: 
> "I used LoRA to make fine-tuning memory-efficient by training only 1% of parameters. I chose rank 64 (higher than typical 8-16) to give the model enough capacity to learn structured JSON output reliably. Targeting both attention and FFN modules ensures the model adapts both its understanding of the input and its generation patterns."

---

### Training Arguments (Hyperparameters)

```python
TrainingArguments(
    # Batch & Accumulation
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,        # Effective batch = 2×4×16 = 128
    
    # Learning Rate
    learning_rate=1e-4,                   # Conservative for LoRA
    warmup_steps=100,                     # Gradual LR ramp-up
    weight_decay=0.01,                    # L2 regularization
    
    # Precision
    bf16=True,                            # BFloat16 training
    fp16=False,                           # Not using FP16
    
    # Optimization
    gradient_checkpointing=True,          # Trade compute for memory
    
    # Data Loading
    dataloader_num_workers=8,             # Parallel data loading
    dataloader_prefetch_factor=2,         # Load 2 batches ahead
    dataloader_persistent_workers=True,   # Don't restart workers each epoch
    
    # Validation & Checkpointing
    eval_strategy="steps",
    eval_steps=50,                        # Validate every 50 steps
    save_steps=50,                        # Save checkpoint every 50 steps
    save_total_limit=3,                   # Keep only 3 checkpoints
    load_best_model_at_end=True,         # Restore best checkpoint
    metric_for_best_model="eval_loss",
    
    # Distributed Training
    ddp_find_unused_parameters=False,     # Optimization for DDP
)
```

---

### GPU Utilization Strategies

#### 1. Gradient Accumulation
```python
per_device_train_batch_size=2
gradient_accumulation_steps=4
# Effective batch size = 2 × 4 × 16 GPUs = 128
```

**Why Small Per-Device Batch Size?**
- Mistral-7B + seq_len=6144 + LoRA adapters = ~55GB per GPU
- batch_size=2 fits in 80GB VRAM with overhead
- Larger batch would cause OOM

**Why Accumulate 4 Steps?**
- Simulates batch_size=8 without OOM
- Larger effective batch improves convergence
- GPU stays busy with compute (no gradient updates every step)

**Technical Explanation**: 
> "I used gradient accumulation to maximize batch size without exceeding GPU memory. Each GPU processes 2 samples per step but accumulates gradients over 4 steps, simulating batch_size=8. The total effective batch of 128 keeps GPUs busy and improves convergence."

---

#### 2. Gradient Checkpointing
```python
gradient_checkpointing=True
```

**What It Does**: Recomputes activations during backward pass instead of storing them

**Trade-off**:
- ✅ 50% less memory (enables longer sequences)
- ❌ ~30% slower training (recomputation overhead)

**Why Worth It**: Allows seq_len=6144 instead of 2048

**Technical Explanation**: 
> "Gradient checkpointing trades compute for memory by recomputing activations during backprop. I accept the 30% slowdown because it lets me fit 6144-token sequences, which is critical for capturing full JSON structures in training data."

---

#### 3. Data Pipeline Optimization
```python
dataloader_num_workers=8                  # Parallel data loading
dataloader_prefetch_factor=2              # Load 2 batches ahead
dataloader_persistent_workers=True        # Don't restart workers
```

**Why This Matters**: GPUs are FAST—data loading can bottleneck training

**How It Works**:
- 8 workers × 2 prefetch = 16 batches ready in CPU memory
- GPUs never wait for data (>95% utilization)
- Persistent workers avoid process creation overhead

**Technical Explanation**: 
> "I parallelized data loading with 8 workers and a prefetch factor of 2 to ensure GPUs never wait for data. This keeps GPU utilization above 80% by eliminating I/O bottlenecks—critical since H100s process batches faster than CPUs can tokenize."

---

#### 4. Mixed Precision Training (BFloat16)
```python
bf16=True
torch_dtype=torch.bfloat16
```

**BF16 vs FP16 vs FP32**:

| Precision | Memory | Speed | Stability | H100 Support |
|-----------|--------|-------|-----------|--------------|
| **FP32** | 100% | 1x | ✅✅✅ | Yes |
| **FP16** | 50% | 2x | ⚠️ (needs loss scaling) | Yes |
| **BF16** | 50% | 2x | ✅✅ | ✅ Native |

**Why BF16 for Training?**
- Same exponent range as FP32 (better numerical stability)
- No loss scaling needed (unlike FP16)
- H100 Tensor Cores accelerate BF16 natively

**Technical Explanation**: 
> "I used BFloat16 mixed precision which H100s accelerate natively. It halves memory usage and doubles throughput while maintaining numerical stability better than FP16. Unlike FP16, BF16 doesn't require loss scaling, making training more reliable."

---

### Validation & Early Stopping

```python
EarlyStoppingCallback(
    early_stopping_patience=8,            # Stop if no improvement for 8 evals
    early_stopping_threshold=0.001,       # Minimum improvement threshold
)

eval_strategy="steps"
eval_steps=50                             # Validate every 50 training steps
```

**Why Validate During Training?**
- Prevents overfitting on small datasets
- Detects training issues early (loss NaN, etc.)
- Enables early stopping to save GPU hours

**Why Patience=8?**
- Too low (3): Might stop before convergence
- Too high (20): Wastes GPU time if model plateaus
- 8 evaluations = ~400 training steps = good balance

**Technical Explanation**: 
> "I implemented validation-based early stopping to prevent overfitting and save GPU resources. By evaluating every 50 steps with patience of 8, I ensure the model stops training once it plateaus, avoiding wasted GPU hours on marginal improvements."

---

## 📊 Training Configuration Summary

### Model & Data
```
Model: Mistral-7B-v0.1 (7B params)
Trainable: ~100M params (LoRA r=64, alpha=128)
Training data: /mnt/shared/data/train.jsonl
Validation data: /mnt/shared/data/val.jsonl
Sequence length: 6144 tokens
```

### Compute Resources
```
GPUs: 16× H100 SXM (2 nodes × 8)
Precision: BFloat16
Memory per GPU: ~45-55GB / 80GB
Training time: 15-30 minutes
```

### Hyperparameters
```
Epochs: 15 (with early stopping)
Batch size: 2 per GPU, 4 accumulation → effective 128
Learning rate: 1e-4 (with 100-step warmup)
Sequence length: 6144 tokens
Optimizer: AdamW with weight_decay=0.01
```

---

## 🚨 Critical Issues & Improvements

### 1. ❌ Flash Attention 2 MISSING (MAJOR BOTTLENECK)

**Current**:
```python
attn_implementation="eager",  # ❌ Slow, memory-hungry
```

**Should Be**:
```python
attn_implementation="flash_attention_2",  # ✅ 2-4x faster
```

**Impact**: Leaving 40-60% GPU utilization on table

**Fix**: `pip install flash-attn --no-build-isolation` and remove eager fallback

**Technical Explanation**: 
> "The biggest bottleneck is missing Flash Attention 2. The code falls back to 'eager' attention which is O(n²) memory and slow. Flash Attention 2 would give 2-4x speedup and 50% memory savings, boosting GPU utilization from ~60% to 90%+. This is in requirements.txt but wasn't installed properly."

---

### 2. ⚠️ Suboptimal Batch Size

**Current**: `per_device_train_batch_size=2` (~55GB used / 80GB available)

**Better**: `per_device_train_batch_size=4` (use full memory)

**Benefit**: ~2x throughput with same memory

**Why Not Done**: Conservative to avoid OOM

---

### 3. ⚠️ No Learning Rate Finder

**Current**: Hard-coded `learning_rate=1e-4`

**Better**: Run LR range test to find optimal rate

**Your Data Might Need**: 5e-5 or 2e-4 instead of 1e-4

---

### 4. ⚠️ Missing DeepSpeed Integration

**Current**: Native PyTorch DDP

**Better**: DeepSpeed ZeRO Stage 2/3

**Benefits**:
- ZeRO-2: Shard optimizer states → 60% less memory per GPU
- Could train larger models or bigger batches

---

### 5. ⚠️ No Gradient Clipping

**Current**: No gradient clipping

**Better**: `max_grad_norm=1.0`

**Why**: Prevents exploding gradients (important with LoRA)

---

### 6. ⚠️ Weak Monitoring

**Current**: `report_to="none"` (no tracking)

**Better**: `report_to=["tensorboard", "wandb"]`

**Missing**: GPU utilization metrics in training logs

---

## 🏆 What You Did RIGHT

✅ **LoRA instead of full fine-tuning** - Huge memory savings  
✅ **Gradient accumulation** - Smart memory/batch size trade-off  
✅ **BF16 mixed precision** - Correct choice for H100s  
✅ **Early stopping** - Prevents overfitting  
✅ **Distributed training setup** - Proper NCCL configuration  
✅ **Data prefetching** - Good I/O optimization  
✅ **Gradient checkpointing** - Memory efficiency for long sequences  
✅ **Validation during training** - Catches issues early  

---

## 🎯 Priority Improvements (High → Low)

### Must Do (Immediate Impact)
1. **Enable Flash Attention 2** → +40% GPU utilization
2. **Increase batch size** → +30% throughput
3. **Add GPU monitoring** → Visibility into bottlenecks
4. **Add gradient clipping** → Training stability

### Should Do (Production Readiness)
5. **DeepSpeed integration** → Better scaling
6. **Learning rate finder** → Better convergence
7. **WandB/TensorBoard** → Better experiment tracking
8. **Checkpoint strategy** → Fault tolerance

### Nice to Have (Optimization)
9. **LoRA target ablation** → Test if all modules needed
10. **Model merging post-training** → Faster inference
11. **Config management** → YAML configs instead of hardcoded

---

## 🎤 Key Technical Responses

### Q: How did you achieve >80% GPU utilization?
**A**: "I optimized the data pipeline with 8 parallel workers and prefetching, used BF16 mixed precision for native H100 acceleration, implemented gradient accumulation to maximize batch size, and enabled gradient checkpointing to fit longer sequences. However, I'm aware Flash Attention 2 is missing—enabling it would boost utilization from ~60% to 90%+."

### Q: Why LoRA instead of full fine-tuning?
**A**: "LoRA reduces trainable parameters by 99%, cutting memory by 95% and training time by 3x while achieving comparable quality. This lets me fine-tune a 7B model on 16 H100s with BF16 precision and long sequences, which wouldn't fit with full fine-tuning."

### Q: How do you prevent overfitting?
**A**: "I use validation-based early stopping with patience of 8 evaluations, weight decay for L2 regularization, and LoRA's inherent regularization through limited expressiveness. The validation set is separate from training, so I can detect overfitting before it becomes severe."

### Q: What's your distributed training strategy?
**A**: "I use PyTorch DDP with torchrun coordinating 16 processes. Each GPU trains on different batches and syncs gradients via NCCL all-reduce over InfiniBand fabric-3. This achieves near-linear scaling—16 GPUs train ~15x faster than 1 GPU. The NCCL configuration enables GPU Direct RDMA to bypass CPUs for inter-node transfers."

---

## 📈 Expected Performance

### GPU Utilization
- **Without Flash Attention**: 50-70%
- **With Flash Attention**: 85-95%
- **Bottleneck**: Attention computation (quadratic complexity)

### Training Time
- **Per epoch**: 60-120 seconds (depending on dataset size)
- **Total (15 epochs)**: 15-30 minutes with early stopping
- **Early stopping typically triggers**: Epoch 8-10

### Memory Usage
- **Per GPU**: ~45-55GB / 80GB available
- **Breakdown**:
  - Model (7B params in BF16): ~14GB
  - LoRA adapters: ~200MB
  - Activations + gradients: ~30GB
  - Optimizer states: ~5GB

### Convergence
- **Initial loss**: ~2.5-3.0
- **Final loss**: ~0.1-0.2
- **Validation loss**: Should track training loss (no overfitting)

---

## 🚀 Conclusion

Your training implementation is **solid with good fundamentals** like LoRA, gradient accumulation, and distributed training. However, **execution leaves performance on the table**:

- **Flash Attention 2 missing** is the biggest issue (40-60% GPU util wasted)
- **Batch size too conservative** (using 55GB of 80GB)
- **Monitoring inadequate** (flying blind on GPU utilization)

With targeted optimizations, this could go from **B+ implementation to A+**. The architecture choices are sound, but performance optimization needs work.
