#!/bin/bash
cd ../

# custom config
DATA="/raid/biplab/hassan/datasets/vqa_abs"
TRAINER=CoOp

DATASET="vqav2"
SEED=1

CFG=vit_b16_ep50


DIR=output/evaluation/${TRAINER}/seed${SEED}
rm -rf output/evaluation/${TRAINER}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"
    # Evaluate on evaluation datasets
    CUDA_VISIBLE_DEVICES=1 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --tpt \

fi