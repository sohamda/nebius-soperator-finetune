# Nebius Soperator Fine-tuning - Copilot Instructions

## Project Overview
Distributed training/fine-tuning and inference on Nebius Kubernetes using Soperator solution.

## Key Tasks
1. Run distributed training/fine-tuning job using [Soperator](https://github.com/nebius/nebius-solution-library/tree/main/soperator)
2. Run inference on the same K8s cluster with the trained model
3. Compare trained vs untrained model results
4. Achieve >80% GPU utilization (monitor via Nebius console dashboards)

## Resource Constraints
- **GPU Limit**: 2x H100s max for training and inference
- **Cluster**: Single MK8s cluster at a time
- **GPU Cluster Resource**: Use only `fabric-3`

## Configuration Requirements
- In `.tfvars` file, set: `public_o11y_enabled = false`
- Install `yq` before running `terraform apply`
- Do NOT share the same filesystem between different jails

## Development Guidelines
- Use async APIs where available for better throughput
- Handle GPU resource allocation carefully to stay within limits
- Monitor GPU utilization via Nebius console dashboards
- Dataset and model choice is flexible

## ML Pipeline
- Dataset is split into train/val/test (80/10/10)
- Training includes validation with early stopping
- Evaluation script measures JSON validity rate (threshold: 80%)
- CI/CD via GitHub Actions (`.github/workflows/ml-pipeline.yml`)

## Project Structure
- `terraform/` - Infrastructure as Code (Soperator, MK8s cluster)
- `training/` - Training scripts and Dockerfile
- `slurm/` - Slurm job scripts for GPU training
- `k8s/` - Kubernetes manifests for inference
- `scripts/` - Dataset generation, evaluation, comparison
- `data/` - Training data (train.jsonl, val.jsonl, test.jsonl)

## Update troubleshooting.md
- Add troubleshoot steps, changes, decesions taken during the task to the md file.
- this will be used to create a knowledge base for future reference and to help others who may face similar issues.