#!/bin/bash

if [[ "$1" == "hybrid" ]]; then
    sbatch test_hybrid.sh 1 brix
    sbatch test_hybrid.sh 1 aweta
    sbatch test_hybrid.sh 1 penetro
elif [[ "$1" == "train" ]]; then
    sbatch training.sh 1 brix
    sbatch training.sh 1 aweta
    sbatch training.sh 1 penetro
fi