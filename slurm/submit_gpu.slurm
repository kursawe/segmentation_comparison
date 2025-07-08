#!/bin/bash
#SBATCH --job-name=segment_w_gpu
#SBATCH --partition=gpu.A30
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --output=/sharedscratch/jk263/slurmlogs/%x_%j.log
#SBATCH --chdir=/sharedscratch/jk263/slurmlogs
#SBATCH --error=/sharedscratch/jk263/slurmlogs/%x_error_%j.log

echo "Running segmentation comparison ..."

source /software/conda/jk263/minforge3/etc/profile.d/conda.sh
conda activate sam2

# time python /sharedscratch/jk263/projects/segmentation_comparison/scripts/cellpose_analysis.py
time python /sharedscratch/jk263/projects/segmentation_comparison/scripts/sam2_analysis.py


