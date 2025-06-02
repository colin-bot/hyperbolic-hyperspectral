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

PYTHONPATH=$PYTHONPATH:../HyperbolicCV/code/:../HyperbolicCV/code/classification/

python3 hybrid_hypcv.py --combined_loss --dataset_label_type brix --n_bins 4 --seed 333 --plot_preds \
    --hyp_weight 0.5 --num_layers 18 --embedding_dim 32 --encoder_manifold euclidean \
    --decoder_manifold lorentz --encoder_k 1.0 --decoder_k 1.0 --pooling_factor 4 --pooling_func min
