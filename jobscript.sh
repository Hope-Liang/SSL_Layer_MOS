#!/bin/bash
#SBATCH -A berzelius-2024-88
#SBATCH --gpus 1
#SBATCH -t 3:00:00

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate py310

DATASET=tencent
MODEL=w2v2_base_tencent_finetuned_10epoch
LY=11
for RUN in {0..4}
do
        python train.py --gin_path configs/$DATASET.gin --save_path results_$DATASET/$MODEL/layer$LY/run$RUN --features_folder $MODEL/layer$LY
done