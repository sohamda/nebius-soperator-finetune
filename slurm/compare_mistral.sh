#!/bin/bash
# =============================================================================
# Mistral-7B Inference Comparison - Base vs Fine-tuned (Two Modes)
# =============================================================================
# Run 1: no-system (base=no sys prompt, finetuned=JSON sys prompt)
# Run 2: with-system (both get identical JSON sys prompt)
# Then combines results into a single report.
# =============================================================================

#SBATCH --job-name=mistral-compare
#SBATCH --output=/mnt/shared/logs/compare_mistral_%j.out
#SBATCH --error=/mnt/shared/logs/compare_mistral_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=main

# =============================================================================
# Environment
# =============================================================================
export HF_HOME=/mnt/shared/cache/huggingface
export CUDA_VISIBLE_DEVICES=0

mkdir -p /mnt/shared/logs

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Inference Comparison: Base vs Fine-tuned Mistral-7B (both modes)"
echo "Start time: $(date)"
echo "======================================================================"

# Monitor GPU utilization in background
nvidia-smi dmon -s u -d 5 > /mnt/shared/logs/gpu_util_compare_mistral_${SLURM_JOB_ID}.log 2>&1 &
GPU_MON_PID=$!

# Activate virtual environment
source /mnt/shared/venv/phi3/bin/activate

cd /mnt/shared/scripts

# =============================================================================
# RUN 1: No system prompt for either (raw model behavior)
# =============================================================================
echo ""
echo "######################################################################"
echo "# RUN 1/2: NO SYSTEM PROMPT (both models, raw behavior)"
echo "######################################################################"
echo ""

python run_inference_compare_mistral.py \
    --checkpoint-dir /mnt/shared/checkpoints/mistral-7b-json-mode \
    --use-4bit \
    --mode no-system \
    --output-tag nosys

# =============================================================================
# RUN 2: Same JSON system prompt for both (fair comparison)
# =============================================================================
echo ""
echo "######################################################################"
echo "# RUN 2/2: JSON SYSTEM PROMPT (both models, same prompt)"
echo "######################################################################"
echo ""

python run_inference_compare_mistral.py \
    --checkpoint-dir /mnt/shared/checkpoints/mistral-7b-json-mode \
    --use-4bit \
    --mode with-system \
    --output-tag withsys

# =============================================================================
# Combined Report
# =============================================================================
echo ""
echo "######################################################################"
echo "# COMBINED REPORT"
echo "######################################################################"
echo ""

python -c "
import json

with open('/mnt/shared/logs/comparison_mistral_nosys.json') as f:
    nosys = json.load(f)
with open('/mnt/shared/logs/comparison_mistral_withsys.json') as f:
    withsys = json.load(f)

print('=' * 70)
print('MISTRAL-7B COMBINED COMPARISON RESULTS')
print('=' * 70)
print()
print(f'{\"\":<35} {\"No Sys Prompt\":<18} {\"JSON Sys Prompt\":<18}')
print('-' * 70)
print(f'{\"Base JSON rate\":<35} {nosys[\"summary\"][\"base_json_rate\"]*100:.0f}%{\"\":>14} {withsys[\"summary\"][\"base_json_rate\"]*100:.0f}%')
print(f'{\"Fine-tuned JSON rate\":<35} {nosys[\"summary\"][\"finetuned_json_rate\"]*100:.0f}%{\"\":>14} {withsys[\"summary\"][\"finetuned_json_rate\"]*100:.0f}%')
print(f'{\"Base avg time\":<35} {nosys[\"summary\"][\"base_avg_time\"]:.1f}s{\"\":>14} {withsys[\"summary\"][\"base_avg_time\"]:.1f}s')
print(f'{\"Fine-tuned avg time\":<35} {nosys[\"summary\"][\"finetuned_avg_time\"]:.1f}s{\"\":>14} {withsys[\"summary\"][\"finetuned_avg_time\"]:.1f}s')
print('=' * 70)
print()
print('Interpretation:')
print('  No Sys Prompt:   Raw model behavior without any JSON instruction.')
print('  JSON Sys Prompt: Both models get the same JSON system message.')
print('  The fine-tuned model should outperform base in both scenarios.')
print('=' * 70)

# Save combined
combined = {
    'no_system_prompt': nosys,
    'with_system_prompt': withsys,
    'combined_summary': {
        'nosys_base_json_rate': nosys['summary']['base_json_rate'],
        'nosys_ft_json_rate': nosys['summary']['finetuned_json_rate'],
        'withsys_base_json_rate': withsys['summary']['base_json_rate'],
        'withsys_ft_json_rate': withsys['summary']['finetuned_json_rate'],
    }
}
with open('/mnt/shared/logs/comparison_mistral_combined.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f'Combined results saved to /mnt/shared/logs/comparison_mistral_combined.json')
"

# Kill GPU monitor
kill $GPU_MON_PID 2>/dev/null

echo "======================================================================"
echo "End time: $(date)"
echo "All comparisons complete!"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  /mnt/shared/logs/comparison_mistral_nosys.json    (no system prompt)"
echo "  /mnt/shared/logs/comparison_mistral_withsys.json   (with JSON system prompt)"
echo "  /mnt/shared/logs/comparison_mistral_combined.json   (combined report)"
echo "  /mnt/shared/logs/gpu_util_compare_mistral_${SLURM_JOB_ID}.log"
