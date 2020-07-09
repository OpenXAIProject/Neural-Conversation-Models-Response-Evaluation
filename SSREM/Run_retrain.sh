#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

python retrain.py --data=$2 --model=$3 --batch_size=$4 --n_epoch=$5 --learning_rate=$6 --checkpoint=$7

wait

echo "Done Run_train.sh $* in $HOSTNAME"
