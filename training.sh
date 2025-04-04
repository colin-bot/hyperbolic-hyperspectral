#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/training_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# python3 train_convnet.py --dataset_label_type median_penetro --n_epochs 30 --lr 0.0001 --classification --special_modes avg1d --seed 99 --plot_preds
# python3 train_convnet.py --dataset_label_type median_penetro --n_epochs 30 --lr 0.0001 --classification --resnet --seed 4 --plot_preds --data_transforms center_crop
# python3 train_convnet.py --dataset_label_type median_penetro --n_epochs 30 --lr 0.00005 --classification --resnet --n_bins 10 --seed 5 --plot_preds --batch_size 16


# python3 train_convnet.py --dataset_label_type penetro --n_epochs 30 --lr 0.00005 --resnet --n_bins 10 --seed 3 --plot_preds

# python3 train_convnet.py --dataset_label_type penetro --n_epochs 30 --lr 0.00005 --classification --resnet --n_bins 8 --seed 3 --eval_only --plot_preds


python3 train_convnet.py --dataset_label_type aweta --n_epochs 30 --lr 0.00001  --classification --resnet --n_bins 8 --seed 69 --pooling_factor 4 --plot_preds

# python3 train_convnet.py --dataset_label_type brix --n_epochs 30 --lr 0.00001 --classification --resnet --n_bins 10 --seed 6 --plot_preds