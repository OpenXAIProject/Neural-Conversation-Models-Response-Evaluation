#!/usr/bin/env bash
# example: bash Run_TC.sh 0 HRED 30 50 True

export CUDA_VISIBLE_DEVICES=$1

python train.py --data=tc_10_10 --model=$2 --batch_size=$3 --eval_batch_size=$3 --n_epoch=$4 --pretrained_wv=$5

wait

