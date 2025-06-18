#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/baseline_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python3 baselines.py --baseline_type linear --dataset_label_type penetro --seed 10 --pca_components 51
python3 baselines.py --baseline_type linear --dataset_label_type brix --seed 10 --pca_components 51
python3 baselines.py --baseline_type linear --dataset_label_type aweta --seed 10 --pca_components 51
