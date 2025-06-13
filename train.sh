#!/bin/bash

#SBATCH --partition=gpu_a100
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

PYTHONPATH=$PYTHONPATH:../deephs_fruit

SEED=$1
LABELTYPE=$2
MODE=$3

if [ -z "$SEED" ]
then
    SEED=1
fi

if [ -z "$LABELTYPE" ]
then
    LABELTYPE=brix
fi

if [ -z "$MODE" ]
then
    MODE=train_euc
fi

NBINS=2

if [[ "$LABELTYPE" == "brix" ]]; then
    NBINS=10
elif [[ "$LABELTYPE" == "aweta" ]]; then
    NBINS=8
elif [[ "$LABELTYPE" == "penetro" ]]; then
    NBINS=8
elif [[ "$LABELTYPE" == "deephs" ]]; then
    NBINS=3
fi

set CUDA_LAUNCH_BLOCKING = 1

if [[ $MODE == "train_euc" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --classification --resnet --seed $SEED
elif [[ $MODE == "train_euc_pooled" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --classification --resnet --seed $SEED --pooling_factor 4 --pooling_func min
elif [[ $MODE == "train_euc_pooled_regr" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --pooling_factor 4 --pooling_func min --plot_preds
elif [[ $MODE == "eval_euc" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --classification --resnet --seed $SEED --eval_only
elif [[ $MODE == "eval_euc_pooled" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --classification --resnet --seed $SEED --eval_only --pooling_factor 4 --pooling_func min
elif [[ $MODE == "combined_loss" ]]; then
    # if [[$SEED == "0"]]; then
    #     python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --plot_preds
    # else
    #     python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED
    # 
    if [[ $LABELTYPE == "all" ]]; then
        python3 train_euc_hypll.py --dataset_label_type brix --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --pca_components 51
        python3 train_euc_hypll.py --dataset_label_type aweta --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --pca_components 51
        python3 train_euc_hypll.py --dataset_label_type penetro --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --pca_components 51
    else
        python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED 
    fi
elif [[ $MODE == "plot_combined_loss" ]]; then 
    if [[ $LABELTYPE == "all" ]]; then
        python3 train_euc_hypll.py --dataset_label_type brix --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --eval_only --plot_preds
        python3 train_euc_hypll.py --dataset_label_type aweta --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --eval_only --plot_preds
        python3 train_euc_hypll.py --dataset_label_type penetro --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --eval_only --plot_preds
    else
        python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --combined_loss --blur_labels --n_bins 8 --n_epochs 30 --lr 0.00001 --resnet --seed $SEED --eval_only --plot_preds
    fi
elif [[ $MODE == "gradcam" ]]; then 
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --classification --resnet --seed $SEED --eval_only --gradcam
elif [[ $MODE == "train_hyp_pooled_regr" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --n_epochs 30 --lr 0.00001 --hypll --seed $SEED --pooling_factor 4 --pooling_func min --plot_preds
elif [[ $MODE == "test_hyp" ]]; then
    python3 train_euc_hypll.py --dataset_label_type median_penetro --n_bins 2 --batch_size 32 --n_epochs 20 --lr 0.001 --hypll --classification --pooling_factor 4 --pooling_func min --seed $SEED
elif [[ $MODE == "train_hscnn" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --batch_size 4 --n_epochs 20 --lr 0.0001 --combined_loss --seed $SEED --eval_only --plot_preds
elif [[ $MODE == "test_pca" ]]; then
    python3 train_euc_hypll.py --dataset_label_type ${LABELTYPE} --n_bins $NBINS --batch_size 4 --n_epochs 20 --lr 0.0001 --combined_loss --resnet --seed $SEED --plot_preds --pca_components 25
fi

# python3 train_euc_hypll.py --dataset_label_type dummy --n_bins 2 --batch_size 32 --n_epochs 20 --lr 0.001 --hypll --classification --pooling_factor 4 --pooling_func min --seed $SEED
