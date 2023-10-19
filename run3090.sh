#!/usr/bin/env bash

export PYTHONPATH=../

python train.py --lr 5e-5 --samp 0.01 --theta 0.1 --worope --gpu 1 --dataset AS12_lag
python train.py --lr 5e-5 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag
python train.py --lr 5e-5 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag
python train.py --lr 5e-5 --samp 0.02 --theta 0.2 --worope --gpu 1 --dataset AS12_lag

python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --worope --gpu 0 --dataset AS12_lag
python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.02 --theta 0.1 --worope --gpu 0 --dataset AS12_lag
python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --epoch 150 --samp 0.01 --theta 0.1 --loss 1 --worope --gpu 0 --dataset AS09_lag
python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --epoch 150 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 0 --dataset AS09_lag
python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --epoch 150 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 0 --dataset AS09_lag
python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --epoch 150 --at ori --worope --gpu 0 --dataset AS09_lag
