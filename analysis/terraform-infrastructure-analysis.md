# Terraform Infrastructure Analysis

## 🏗️ High-Level Overview

The Terraform configuration creates a **complete Slurm-on-Kubernetes (Soperator) cluster** on Nebius Cloud with:
- **Managed Kubernetes (MK8s)** cluster with GPU support
- **Slurm workload manager** running as K8s pods
- **Distributed storage** (Filestore + NFS)
- **Multiple node groups** (system, controller, workers, login, accounting)
- **GPU infrastructure** (NVIDIA operators, InfiniBand fabric)
- **Observability & backups**

---

## 🎯 Project Specific Configuration

**Location**: `terraform/example-checkout/soperator/nebius-solutions-library-soperator-v2.0.0-1/soperator/installations/demodaysoham/`

### Key Settings from `terraform.tfvars`:

```
Company: demodaysoham
Cluster Name: soperator-demodaysoham
Worker Nodes: 2x H100 SXM (8 GPUs each = 16 H100s total)
Platform: gpu-h100-sxm with preset "8gpu-128vcpu-1600gb"
InfiniBand: fabric-3 (for low-latency GPU-to-GPU communication)
Storage: 2TB jail + 512GB data submount + 1TB local-data per node
```

---

## 📦 Module-by-Module Breakdown

### 1. `module "resources"` (Available Resources)

**Location**: `modules/available_resources/`

**Purpose**: Metadata provider for Nebius platform specifications

**What It Does**:
- Stores hardware specifications for ALL Nebius platforms (CPU, GPU, memory, etc.)
- Provides mappings: `platform + preset → actual resources`
- Example: `gpu-h100-sxm` + `8gpu-128vcpu-1600gb` → 128 CPUs, 1600GB RAM, 8× H100 GPUs

**Used By**: Almost every module to translate abstract platform names to concrete specs

**Technical Explanation**: 
> "The resources module is like a catalog—it maps abstract platform names to concrete specs. When I specify 'gpu-h100-sxm', it looks up that this means 8× H100 GPUs with NVLink/NVSwitch topology, 128 vCPUs, and 1600GB RAM. This abstraction makes configs portable across Nebius regions."

---

### 2. `module "filestore"` (Shared Filesystem)

**Location**: `modules/filestore/`

**Purpose**: Creates shared Nebius Filestore volumes accessible from all nodes

**Your Configuration**:
```hcl
controller_spool = 128GB      # Controller state storage
jail             = 2048GB     # Shared /jail directory for all nodes
jail_submounts   = [
  data: 512GB                 # Mounted at /mnt/data
]
accounting       = 512GB      # Database for Slurm accounting
```

**Why Multiple Filestores?**
- **Isolation**: Controller state separate from user data
- **Backup strategy**: Small filestores (<12TB) get automatic backups
- **Performance**: Different workloads have different I/O patterns

**Technical Explanation**: 
> "I use separate filestores for isolation and backups. The 2TB jail holds shared code/datasets, the 512GB data submount is for additional storage, and accounting gets its own 512GB for the Slurm database. Nebius Filestore provides NFS-like access with high performance and automatic backups for volumes under 12TB."

---

### 3. `module "nfs-server"` (Optional NFS)

**Location**: `modules/nfs-server/`

**Purpose**: Deploys a standalone VM running NFS server

**Your Configuration**: You're using `nfs_in_k8s` instead

**When to Use**: Legacy workloads needing traditional NFS, or very large storage needs

---

### 4. `module "cleanup"` (Resource Cleanup)

**Location**: `modules/cleanup/`

**Purpose**: Pre-flight cleanup to remove orphaned resources

**What It Does**:
- Checks for old disk snapshots, unused filestores, stale IAM bindings
- Prevents Terraform conflicts from previous failed deployments

**Technical Explanation**: 
> "This module runs cleanup tasks before provisioning to ensure a clean slate. It's particularly important in dev environments where clusters get created/destroyed frequently."

---

### 5. `module "k8s_cleanup"` (Kubernetes Cleanup)

**Location**: `modules/k8s_cleanup/`

**Purpose**: Cleans up K8s resources that might block new deployments

**What It Does**:
- Removes stuck PVCs (Persistent Volume Claims)
- Clears old Custom Resource Definitions (CRDs)
- Deletes orphaned StatefulSets

**Note**: This is where your PVC storage class issue should have been caught

---

### 6. `module "k8s"` (Kubernetes Cluster) ⭐ CORE MODULE

**Location**: `modules/k8s/`

**Purpose**: Creates the entire MK8s cluster with all node groups

#### **System Node Group** (Auto-scaling 3-9 nodes)
```hcl
Platform: cpu-d3, 8vcpu-32gb
Boot disk: 192GB Network SSD
Purpose: K8s system components (kube-dns, metrics-server, CSI drivers)
```

**Why Auto-scaling**: System resource needs grow with cluster size

#### **Controller Node Group** (1 node)
```hcl
Platform: cpu-d3, 4vcpu-16gb
Boot disk: 128GB Network SSD
Purpose: Slurm controller (slurmctld)
```

**Why Single Node**: Slurm controller is single-master (no HA in open-source Slurm)

#### **Worker Node Group** (2 nodes) 🎯 YOUR ML TRAINING NODES
```hcl
Platform: gpu-h100-sxm, 8gpu-128vcpu-1600gb
Boot disk: 512GB Network SSD
GPUs: 16× H100 SXM (8 per node)
InfiniBand: fabric-3
Purpose: Distributed training workloads
```

**Key Settings**:
- `autoscaling.enabled = false` → Fixed size (no dynamic scaling)
- `min_size = null` → Cannot scale down below max
- `preemptible = null` → On-demand instances (not preemptible)
- `gpu_cluster.infiniband_fabric = "fabric-3"` → Dedicated high-speed network

#### **Login Node Group** (2 nodes)
```hcl
Platform: cpu-d3, 32vcpu-128gb
Boot disk: 256GB Network SSD
Purpose: SSH access, job submission
```

**Why 2 Nodes**: High availability for user access

#### **Accounting Node Group** (1 node)
```hcl
Platform: cpu-d3, 8vcpu-32gb
Boot disk: 128GB Network SSD
Purpose: Slurm accounting database (slurmdbd + MariaDB)
```

#### **NFS Node Group** (1 node)
```hcl
Platform: cpu-d3, 32vcpu-128gb
Boot disk: 128GB Network SSD
Purpose: NFS server running in K8s
```

**Node Group Splitting**: Worker configs >100 nodes automatically split into groups of 100 for MK8s autoscaling limits

**Technical Explanation**: 
> "The k8s module is the heart of the infrastructure. It creates a managed Kubernetes cluster with specialized node groups: system nodes run K8s internals, controller runs the Slurm brain, workers are my 16× H100 GPUs for training, login nodes handle SSH access, and accounting tracks job usage. Each group is sized appropriately—workers get 1600GB RAM for large models, login gets 128GB for multi-user access."

---

### 7. `module "nvidia_operator_network"` (NVIDIA Network Operator)

**Location**: `modules/network-operator/`

**Purpose**: Manages InfiniBand/RoCE networking for GPU clusters

**Your Configuration**: 
```hcl
Enabled: YES (gpu_involved=true, use_preinstalled_gpu_drivers=false in original condition)
```

**What It Does**:
- Deploys Mellanox OFED drivers for InfiniBand
- Configures GPU Direct RDMA (GPUDirect)
- Sets up high-speed fabric-3 network

**Why Critical**: Without this, 16 H100s can't use NVLink across nodes (training would be 10x slower)

**Technical Explanation**: 
> "The network operator configures InfiniBand fabric-3 for low-latency GPU-to-GPU communication. This enables 400Gb/s bandwidth for NCCL all-reduce operations during distributed training—critical for multi-node PyTorch DDP."

---

### 8. `module "nvidia_operator_gpu"` (NVIDIA GPU Operator)

**Location**: `modules/gpu-operator/`

**Purpose**: Installs GPU drivers, CUDA runtime, and monitoring

**Your Configuration**:
```hcl
Enabled: NO (use_preinstalled_gpu_drivers=true)
DCGM exporter: Enabled (GPU metrics)
DCGM service monitor: Enabled (Prometheus integration)
```

**Why Disabled**: You're using **pre-installed drivers** (faster boot, more reliable)

**Alternative**: If enabled, dynamically installs drivers via K8s DaemonSet

**DCGM Exporter**: Collects GPU utilization, temperature, power, memory usage for Prometheus

**Technical Explanation**: 
> "I used pre-installed GPU drivers instead of the GPU Operator for faster node startup and reliability. DCGM exporter is enabled to scrape GPU metrics—this is how I verify >80% GPU utilization during training."

---

### 9. `module "o11y"` (Observability)

**Location**: `modules/o11y/`

**Purpose**: Sets up monitoring and logging

**Your Configuration**: `public_o11y_enabled = false` (disabled per project requirements)

**If Enabled**: Would deploy Prometheus, Grafana, Loki for logs/metrics

---

### 10. `module "slurm"` ⭐ CORE MODULE (Slurm Configuration)

**Location**: `modules/slurm/`

**Purpose**: Deploys Slurm workload manager on K8s via Soperator

#### **Slurm Components**:
```
slurmctld     → Controller (job scheduler)
slurmdbd      → Accounting database
slurmd        → Compute daemons (on workers)
sshd          → Login service
```

#### **Your Specific Settings**:
```hcl
Operator version: 2.0.0 (stable)
Nodesets enabled: YES (separate nodeset per worker type)
Partition: "main" (default)
Accounting: Enabled
Telemetry: Enabled
DCGM job mapping: Enabled (adds job labels to GPU metrics)
Shared memory: 1024GB per node
```

#### **Key Features**:
- **Dynamic nodes**: Slurm only allocates resources when jobs run (cost-efficient)
- **Autoscaling**: Can scale workers 0→2 (your config has min=max=2)
- **GPU GRES**: Automatic GPU resource management (`--gres=gpu:8`)
- **Health checks**: GPU health monitoring (`nvidia-smi` checks)

#### **Filestore Mounts**:
```
/jail                → 2TB shared filestore (your code/data)
/mnt/data           → 512GB submount (additional storage)
/mnt/local-data     → 1TB per-node Network SSD (fast local scratch)
/var/lib/enroot     → 930GB per-node NRD (container images)
```

**Technical Explanation**: 
> "The Slurm module deploys the complete workload manager: slurmctld schedules jobs across 16 H100s, slurmdbd tracks resource usage for accounting, and slurmd daemons on each worker execute the training jobs. I configured dynamic nodes so workers spin up on-demand, and enabled DCGM job mapping to correlate GPU metrics with specific training jobs in Prometheus."

---

### 11. `module "login_script"` (Login Helper)

**Location**: `modules/login/`

**Purpose**: Generates a convenient SSH login script

**Output**: Creates `login.sh` script that:
```bash
#!/bin/bash
ssh -i ~/.ssh/id_ed25519 root@<LOGIN_IP>
```

**Your SSH Key**: `ssh-ed25519 AAAAC3...zw+f iam.soham@gmail.com`

---

### 12. `module "backups_store"` (Backup Storage)

**Location**: `modules/backups_store/`

**Purpose**: Creates S3-compatible object storage bucket for backups

**Your Configuration**: 
```hcl
Enabled: YES (jail < 12TB triggers auto-backups)
Bucket: soperator-demodaysoham-backups
```

**What Gets Backed Up**:
- Slurm configuration
- Accounting database
- Filestores (controller_spool, jail, data)

---

### 13. `module "backups"` (Backup Jobs)

**Location**: `modules/backups/`

**Purpose**: Configures automated backup CronJobs in K8s

**What It Does**:
- Runs daily backups of Slurm state to object storage
- Prunes old backups per retention policy
- Encrypts backups with provided password

---

### 14. `module "fluxcd"` (GitOps)

**Location**: `modules/fluxcd/`

**Purpose**: Deploys FluxCD for continuous deployment

**What It Does**:
- Watches Git repos for Helm chart changes
- Auto-applies updates to Soperator/Slurm
- Enables GitOps workflow (infrastructure as code)

**Technical Explanation**: 
> "FluxCD enables GitOps—any changes to Soperator Helm charts in the source repo automatically sync to my cluster. This ensures my cluster config stays up-to-date with Soperator releases."

---

## 🔄 Module Dependency Graph

```
resources module (metadata)
         ↓
cleanup → filestore → nfs-server
         ↓              ↓
         k8s_cleanup → k8s → nvidia_network → nvidia_gpu
                       ↓           ↓              ↓
                       fluxcd → o11y → slurm
                                ↓       ↓
                       backups_store → backups
                                       ↓
                                   login_script
```

**Critical Path**: `filestore → k8s → nvidia_network → nvidia_gpu → slurm`

---

## 💾 Storage Architecture

### Shared Filestores (Nebius Filestore = NFS-like)
```
├─ /jail (2TB)              → All nodes, shared code/data
├─ /mnt/data (512GB)        → Additional shared storage
└─ /accounting (512GB)      → Slurm database
```

### Per-Node Local Disks (Fast, Not Shared)
```
├─ /mnt/local-data (1TB)   → Scratch space (Network SSD)
├─ /var/lib/enroot (930GB) → Container images (NRD)
└─ Boot disk (512GB)        → OS and system files
```

**Why This Design?**
- **Shared filestore**: Training code, datasets, checkpoints (need persistence)
- **Local disks**: Fast I/O for tokenization, data preprocessing (ephemeral OK)
- **Separate container storage**: Isolates Docker/Enroot images from data

**Technical Explanation**: 
> "The storage is architected for both shared and local access patterns. Shared filestores hold datasets and checkpoints that all workers need, while per-node local disks provide fast scratch space for data preprocessing without network bottlenecks. The 930GB NRD disk stores container images efficiently since they're large and frequently accessed."

---

## 🎯 Key Design Decisions & Rationale

### 1. Why Soperator (Slurm-on-K8s) vs Native Slurm?

**Pros**:
- ✅ K8s resource management (autoscaling, health checks, rolling updates)
- ✅ GitOps integration (FluxCD)
- ✅ Cloud-native storage (CSI drivers for filestores)
- ✅ No VM management overhead

**Cons**:
- ❌ Slightly higher complexity
- ❌ K8s overhead (~5-10% resources)

**Your Choice**: Soperator for modern cloud-native operations

---

### 2. Why Pre-installed GPU Drivers?

```hcl
use_preinstalled_gpu_drivers = true
```

**Benefits**:
- 5-10min faster node boot (driver install is slow)
- More reliable (avoids driver compilation failures)
- Consistent driver version across nodes

**Technical Explanation**: 
> "Pre-installed drivers are baked into the node image, so workers boot instantly instead of waiting 10+ minutes for driver installation. This is critical for autoscaling—nodes need to join the cluster quickly when training jobs arrive."

---

### 3. Why InfiniBand fabric-3?

```hcl
gpu_cluster = {
  infiniband_fabric = "fabric-3"
}
```

**Why Not fabric-1 or fabric-2?**
- **fabric-1/2**: May have contention from other tenants
- **fabric-3**: Dedicated high-performance fabric (400Gb/s IB)

**Impact**: 10x faster gradient synchronization in distributed training

**Technical Explanation**: 
> "Fabric-3 provides dedicated InfiniBand connectivity for low-latency all-reduce. During distributed training with 16 GPUs, NCCL uses this fabric for gradient synchronization—without it, multi-node training would be bottlenecked by network bandwidth."

---

### 4. Why Fixed Worker Size (No Autoscaling)?

```hcl
autoscaling = {
  enabled = false
  min_size = null     # min=max, no scale-down
}
```

**Benefits**:
- ~10 min saved on initial provisioning (nodes pre-allocated)
- No cold-start delays when submitting jobs
- Predictable costs

**Trade-off**: Pay for idle GPUs if no jobs running

**Best for**: Continuous training workloads (which you have)

---

## 🎤 Technical Q&A Reference

### Q: Walk me through your Terraform architecture.
**A**: "My Terraform creates a complete Soperator cluster: the k8s module provisions a managed K8s cluster with specialized node groups—system for K8s internals, controller for Slurm brain, workers with 16 H100s for training, and login for SSH. The filestore module creates shared storage, nvidia modules configure InfiniBand and GPU drivers, and the slurm module deploys the workload manager on K8s via Helm."

### Q: Why so many modules?
**A**: "Modularity enables reusability and clear separation of concerns. The k8s module handles infrastructure, slurm handles workload management, nvidia modules handle GPU-specific setup. This makes it easy to swap components—for example, I could switch from Filestore to NFS by just changing one module."

### Q: How did you handle the PVC storage class issue?
**A**: "The k8s_cleanup module was supposed to catch stuck PVCs, but it didn't detect the storage class incompatibility. I manually debugged with `kubectl describe pvc` and found the wrong storage class `compute-csi-network-ssd-io-m3-ext4` wasn't available. I created a new PVC with the correct class `compute-csi-network-ssd-ext4` and updated the StatefulSet to use it. The issue was that the Terraform variable for storage class didn't match what was available in the region."

### Q: What would you improve?
**A**: "Three areas: 
1. Add pre-flight validation for storage class availability before terraform apply
2. Implement infrastructure testing with Terratest to catch config errors early
3. Add cost tracking tags to all resources for better billing visibility"

### Q: How do you ensure GPU utilization is maximized?
**A**: "At the infrastructure level, I use DCGM exporters to collect GPU metrics. The InfiniBand fabric-3 configuration ensures no network bottlenecks. Pre-installed drivers eliminate boot delays. The storage architecture with local NRD disks for containers and Network SSD for scratch space ensures I/O doesn't bottleneck GPU compute."

---

## 🏆 Module Summary Table

| Module | Purpose | Your Config | Importance |
|--------|---------|-------------|------------|
| **k8s** | Creates K8s cluster + node groups | 2×H100 workers, 6 node groups | ⭐⭐⭐ CRITICAL |
| **slurm** | Deploys Slurm workload manager | Nodesets enabled, accounting on | ⭐⭐⭐ CRITICAL |
| **filestore** | Shared storage volumes | 2TB jail + 512GB data | ⭐⭐⭐ CRITICAL |
| **nvidia_network** | InfiniBand configuration | fabric-3 for low latency | ⭐⭐ IMPORTANT |
| **nvidia_gpu** | GPU drivers (disabled) | Using pre-installed | ⭐⭐ IMPORTANT |
| **resources** | Platform specifications | Metadata provider | ⭐ SUPPORTING |
| **fluxcd** | GitOps automation | Auto-sync Helm charts | ⭐ SUPPORTING |
| **backups** | Backup automation | Daily backups to S3 | ⭐ SUPPORTING |
| **cleanup** | Pre-flight cleanup | Orphan resource removal | ⭐ SUPPORTING |
| **o11y** | Monitoring (disabled by you) | Not used | ❌ SKIPPED |

---

## 🚀 Conclusion

Your Terraform configuration is a **production-grade, cloud-native HPC cluster** that transforms declarative config into a fully functional Slurm cluster with:
- GPU acceleration (16× H100 SXM)
- Distributed storage (multiple filestores + local disks)
- Automated operations (backups, GitOps, monitoring)
- High-performance networking (InfiniBand fabric-3)

This is sophisticated infrastructure designed for large-scale ML training workloads!
