#!/bin/bash

#SBATCH --partition=gpu_a100
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

SEED=$1
LABELTYPE=$2

if [ -z "$SEED" ]
then
    SEED=1
fi

if [ -z "$LABELTYPE" ]
then
    LABELTYPE=brix
fi

if [[ "$LABELTYPE" == "brix" ]]; then
    NBINS=10
elif [[ "$LABELTYPE" == "aweta" ]]; then
    NBINS=8
elif [[ "$LABELTYPE" == "penetro" ]]; then
    NBINS=8
fi

if [[ $SEED == "1" ]]; then
    python3 hybrid_test.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --classification --seed 1 --plot_preds --hyp_weight 0.5
elif [[ $SEED == "2" ]]; then
    python3 hybrid_test.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --classification --seed 2 --pooling_func min --pooling_factor 4 --plot_preds --hyp_weight 0.5
fi
