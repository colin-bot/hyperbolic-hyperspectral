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
    sbatch training.sh 1 brix train_euc 
    sbatch training.sh 1 aweta train_euc
    sbatch training.sh 1 penetro train_euc
elif [[ "$1" == "eval_euc" ]]; then
    sbatch training.sh 2 brix eval_euc_pooled
    sbatch training.sh 2 aweta eval_euc_pooled
    sbatch training.sh 2 penetro eval_euc_pooled
fi