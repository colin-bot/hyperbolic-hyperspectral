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

for label in "brix" "aweta" "penetro" 
do
        python3 train_hypcv.py --n_epochs 10 --lr 0.001 --loss_weights 0.01-1.0-0.1 --seed 0 --dataset_label_type $label \
        --n_bins 8 --combined_loss --blur_labels --plot_preds --num_layers 18 --embedding_dim 32 \
        --encoder_manifold lorentz --decoder_manifold euclidean --encoder_k 1.0 --decoder_k 1.0 \
        --use_lr_scheduler --optimizer RiemannianAdam
done


