#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hybrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/hybrid_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python3 hybrid_test.py --dataset_label_type brix  --n_bins 10 --classification --seed 1 --plot_preds
