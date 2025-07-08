#!/bin/bash
#SBATCH --job-name=segment_stuff
#SBATCH --partition=gpu.A100
#SBATCH --nodes=1
#SBATCH --ntasks=1 # up to 168 on one node
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --output=/sharedscratch/jk263/slurmlogs/%x_%j.log

echo "Running segmentation comparison ..."

source /software/conda/jk263/minforge3/etc/profile.d/conda.sh
conda activate sam2

time python /sharedscratch/jk263/projects/segmentation_comparison/scripts/cellpose_analysis.py
time python /sharedscratch/jk263/projects/segmentation_comparison/scripts/sam2_analysis.py

