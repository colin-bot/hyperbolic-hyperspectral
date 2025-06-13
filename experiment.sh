#!/bin/bash

if [[ "$1" == "hybrid" ]]; then
    sbatch test_hybrid.sh 1 brix
    sbatch test_hybrid.sh 1 aweta
    sbatch test_hybrid.sh 1 penetro
elif [[ "$1" == "hybrid2" ]]; then
    sbatch test_hybrid.sh 2 brix
    sbatch test_hybrid.sh 2 aweta
    sbatch test_hybrid.sh 2 penetro
elif [[ "$1" == "train_euc" ]]; then
    sbatch train.sh 1 brix train_euc 
    sbatch train.sh 1 aweta train_euc
    sbatch train.sh 1 penetro train_euc
elif [[ "$1" == "eval_euc" ]]; then
    sbatch train.sh 2 brix eval_euc_pooled
    sbatch train.sh 2 aweta eval_euc_pooled
    sbatch train.sh 2 penetro eval_euc_pooled
elif [[ "$1" == "combined_loss" ]]; then
    for seed in "0"
    # for seed in "0" "1" "2" "3" "4"
    do
        sbatch train.sh $seed all combined_loss
    done
fi
