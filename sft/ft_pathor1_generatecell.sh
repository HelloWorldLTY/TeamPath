#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu_h200
#SBATCH --time=24:00:00
#SBATCH --job-name=pytorch
### GPU options ###
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=verbose,closest

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

module load CUDA
DISABLE_VERSION_CHECK=1 CUDA_LAUNCH_BLOCKING=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_patho.yaml

# module load CUDA
# CUDA_LAUNCH_BLOCKING=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml

# module load CUDA
# DISABLE_VERSION_CHECK=1 CUDA_LAUNCH_BLOCKING=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_patho_summary.yaml