#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_output/download_%A.out

module purge
module load 2023

# wget -O /scratch-shared/cbot/Avocado.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Avocado.zip
# wget -O /scratch-shared/cbot/Kaki.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kaki.zip
# wget -O /scratch-shared/cbot/Kiwi.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kiwi.zip
# wget -O /scratch-shared/cbot/Mango.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Mango.zip
# wget -O /scratch-shared/cbot/Papaya.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Papaya.zip

# wget -O /scratch-shared/cbot/annotations.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations.zip
wget -O /scratch-shared/cbot/annotations-upd-2024-01-09.zip https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations-upd-2024-01-09.zip

# wget -O /scratch-shared/cbot/readme.txt https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/readme.txt
# wget -O /scratch-shared/cbot/annotations-upd-2024-01-09.ReadMe https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations-upd-2024-01-09.ReadMe

# unzip /scratch-shared/cbot/Avocado.zip -d /scratch-shared/cbot
# unzip /scratch-shared/cbot/Kaki.zip -d /scratch-shared/cbot
# unzip /scratch-shared/cbot/Kiwi.zip -d /scratch-shared/cbot
# unzip /scratch-shared/cbot/Mango.zip -d /scratch-shared/cbot
# unzip /scratch-shared/cbot/Papaya.zip -d /scratch-shared/cbot
# unzip /scratch-shared/cbot/annotations.zip -d /scratch-shared/cbot
# mkdir /scratch-shared/cbot/anno2
# mv /scratch-shared/cbot/annotations-upd-2024-01-09.zip /scratch-shared/cbot/anno2/
# unzip /scratch-shared/cbot/anno2/annotations-upd-2024-01-09.zip -d /scratch-shared/cbot/anno2/
# mv /scratch-shared/cbot/anno2/annotations /scratch-shared/cbot/anno2/annotations-upd-2024-01-09
# mv /scratch-shared/cbot/anno2/annotations-upd-2024-01-09 /scratch-shared/cbot/annotations-upd-2024-01-09
# rm -rf /scratch-shared/cbot/anno2/

rm /scratch-shared/cbot/Avocado.zip
rm /scratch-shared/cbot/Kaki.zip
rm /scrtch-shared/cbot/Kiwi.zip
rm /scratch-shared/cbot/Mango.zip
rm /scratch-shared/cbot/Papaya.zip
rm /scratch-shared/cbot/annotations.zip
rm /scratch-shared/cbot/annotations-upd-2024-01-09.zip
