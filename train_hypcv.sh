#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=trainhyp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/train_hyp_%A.out

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

PYTHONPATH=$PYTHONPATH:../HyperbolicCV/code/:../HyperbolicCV/code/classification/:../deephs_fruit

# python3 train_convnet.py --n_epochs 30 --lr 0.005 --seed 1 --dataset_label_type penetro \
#         --n_bins 4 --combined_loss --blur_labels --plot_preds --pooling_factor 4 --pooling_func min

python3 train_hypcv.py --n_epochs 30 --lr 0.001 --seed 333 --dataset_label_type brix \
        --n_bins 8 --combined_loss --blur_labels --plot_preds --num_layers 18 --embedding_dim 32 \
        --encoder_manifold lorentz --decoder_manifold lorentz --encoder_k 1.0 --decoder_k 1.0 \
        --use_lr_scheduler --optimizer RiemannianAdam --pooling_factor 4 --pooling_func min
# python3 train_hypcv.py --n_epochs 20 --lr 0.0001 --seed 1 --dataset_label_type aweta --n_bins 8 --combined_loss --blur_labels --plot_preds --num_layers 18 --embedding_dim 32 --encoder_manifold lorentz --decoder_manifold lorentz --encoder_k 1.0 --decoder_k 1.0


