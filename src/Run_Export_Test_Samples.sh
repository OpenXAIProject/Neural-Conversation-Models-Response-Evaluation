#!/usr/bin/env bash
# example: bash Run_Export_Test_Samples.sh HRED 30 True 1 1 30.pkl

export CUDA_VISIBLE_DEVICES=$1

python export_test_responses.py --data=tc_10_10 --model=$2 --batch_size=$3 --pretrained_wv=$4 --n_context=$5 --n_sample_step=$6 --checkpoint=$7

wait

