#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/training_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# python3 train_convnet.py --dataset_label_type brix --n_epochs 30 --lr 0.00001  --classification --resnet --n_bins 10 --seed 7 --plot_preds
python3 train_convnet.py --dataset_label_type brix --n_epochs 30 --lr 0.00001 --classification --resnet --n_bins 10 --seed 69 --pooling_factor 4 --pooling_func min --plot_preds

# python3 train_convnet.py --dataset_label_type aweta --classification --n_bins 8 --n_epochs 5 --lr 0.01 --hypll --seed 1 --plot_preds 
# python3 train_convnet.py --dataset_label_type aweta --classification --n_bins 8 --n_epochs 5 --lr 0.01 --hypll --seed 2 --pooling_factor 4 --pooling_func min --plot_preds
