#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1

python eval1.py --data=$2 --model=$3 --batch_size=$4 --checkpoint=$5 --test_target_ng=$6

wait

echo "Done Run_Eval1.sh $* in $HOSTNAME"
