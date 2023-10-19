#!/usr/bin/env bash

export PYTHONPATH=../

# ASSISTments_09-10

#python train.py --lr 5e-4 --samp 0.02  --theta 0.1 --gpu 1 0.8434
#python train.py --lr 5e-4 --samp 0.02  --theta 0.1 --SENet --gpu 0
#python train.py --lr 5e-4 --samp 0.02  --theta 0.2 --SENet --gpu 0
#python train.py --lr 5e-4 --samp 0.01  --theta 0.1 --SENet --gpu 0
#python train.py --lr 5e-4 --samp 0.01  --theta 0.2 --SENet --gpu 0
#python train.py --lr 5e-4 --samp 0.04  --theta 0.1 --SENet --gpu 0
#python train.py --lr 5e-4 --samp 0.04  --theta 0.2 --SENet --gpu 0


#python train.py --lr 5e-4 --samp 0.02  --theta 0.1 --loss 1 --ln 2 --gpu 0
#python train.py --lr 5e-4 --samp 0.02  --theta 0.1 --loss 2 --ln 2 --gpu 0
#python train.py --lr 5e-4 --samp 0.02  --theta 0.1 --loss 3 --ln 2 --gpu 0
#python train.py --lr 5e-4 --samp 0.02  --theta 0 --loss 3 --SENet --gpu 0


#python train.py --lr 5e-4 --samp 0.02  --theta 0.1 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.01  --theta 0.1 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.02  --theta 0.2 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.01  --theta 0.2 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --at ori --gpu 0 --dataset AS09_lag

#python train.py --lr 5e-4 --samp 0.02  --theta 0.2  --loss 1 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.02  --theta 0.2  --loss 2 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.02  --theta 0.2  --loss 3 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --at ori --gpu 0 --dataset AS09_lag --worope
#python train.py --lr 5e-4 --at ori --gpu 0 --dataset AS09_lag --wob
#
#python train.py --lr 5e-4 --samp 0.02  --theta 0.3 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.02  --theta 0.4 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.05  --theta 0.1 --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.05  --theta 0.2 --gpu 0 --dataset AS09_lag

#python train.py --lr 5e-4 --samp 0.02  --theta 0.2 --gpu 0 --dataset AS12_lag --worope
#python train.py --lr 5e-4 --samp 0.02  --theta 0.2 --gpu 0 --dataset AS12_lag --wob
#python train.py --lr 5e-4 --at ori --gpu 0 --dataset AS09_lag

#python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --worope --gpu 0 --dataset AS12_lag
#python train.py --lr 5e-4 --samp 0.01 --theta 0.2 --worope --gpu 0 --dataset AS12_lag
#python train.py --lr 5e-4 --samp 0.02 --theta 0.1 --worope --gpu 0 --dataset AS12_lag
#python train.py --lr 5e-4 --samp 0.02 --theta 0.2 --worope --gpu 0 --dataset AS12_lag
#
#python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.01 --theta 0.2 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.02 --theta 0.1 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.02 --theta 0.2 --worope --gpu 0 --dataset AS09_lag

#python train.py --lr 5e-4 --at ori --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --loss 1 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 0 --dataset AS09_lag
#
#python train.py --lr 5e-5 --at ori --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-5 --samp 0.01 --theta 0.1 --loss 1 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-5 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-5 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 0 --dataset AS09_lag

#python train.py --lr 5e-4 --epoch 150 --at ori --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --epoch 150 --samp 0.01 --theta 0.1 --loss 1 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --epoch 150 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 0 --dataset AS09_lag
#python train.py --lr 5e-4 --epoch 150 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 0 --dataset AS09_lag


#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --loss 1 --worope --gpu 0 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 1 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 1 --dataset AS12_lag



#python train.py --lr 5e-4 --adp 0 --fdp 0 --samp 0.01 --theta 0.1 --ln 2 --loss 1 --worope --gpu 1 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0 --fdp 0 --samp 0.01 --theta 0.1 --ln 2 --loss 2 --worope --gpu 1 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0 --fdp 0 --samp 0.01 --theta 0.1 --ln 2 --loss 3 --worope --gpu 1 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --ln 2 --loss 1 --worope --gpu 1 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --ln 2 --loss 2 --worope --gpu 1 --dataset AS12_lag
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --ln 2 --loss 3 --worope --gpu 1 --dataset AS12_lag



#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --at ori --worope --gpu 1 --dataset AS12_lag2  --batch_size 256 --epoch 100
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --at ori --worope --gpu 1 --dataset AS12_lag2  --batch_size 128 --epoch 100
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --at ori --worope --gpu 1 --dataset AS12_lag2 --epoch 100
#
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag2  --batch_size 256  --epoch 180
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag2  --batch_size 128  --epoch 180
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.3 --worope --gpu 1 --dataset AS12_lag2  --batch_size 128  --epoch 180
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --samp 0.02 --theta 0.2 --worope --gpu 1 --dataset AS12_lag2  --batch_size 128  --epoch 180
#
#python train.py --lr 1e-3 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --worope --gpu 1 --dataset AS12_lag2  --epoch 180
#python train.py --lr 1e-3 --adp 0.1 --fdp 0.1 --samp 0.01 --theta 0.1 --worope --gpu 1 --dataset AS12_lag2  --batch_size 128  --epoch 180
#python train.py --lr 1e-3 --adp 0.1 --fdp 0.1 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag2  --epoch 180

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 256 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 256 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 1e-3 --adp 0.1 --fdp 0.1 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 15
#python train.py --lr 1e-3 --adp 0.1 --fdp 0.1 --encoders 5 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 15

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.01 --theta 0.1 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.01 --theta 0.2 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.01 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.02 --theta 0.1 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.02 --theta 0.2 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.02 --theta 0.1 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.02 --theta 0.1 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.2 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.2 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.02 --theta 0.1 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.02 --theta 0.1 --worope --wof --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --at ori --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --at ori --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --at ori --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --at ori --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --at ori --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 30

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.1 --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.02 --theta 0.1 --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.2 --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.02 --theta 0.2 --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.02 --theta 0.2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.1 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.1 --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.01 --theta 0.1 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.01 --theta 0.1 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.1 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.2 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --samp 0.01 --theta 0.2 --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.01 --theta 0.2 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --samp 0.01 --theta 0.2 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.2 --wof --worope --gpu 0 --dataset AS12_lag3  --batch_size 128 --epoch 80

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.04 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.05 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --batch_size 256 --samp 0.02 --theta 0.1 --worope --gpu 1 --dataset AS12_lag3 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.03 --theta 0.2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 4 --ln 2 --samp 0.02 --theta 0.2 --wof --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.1 --loss 1 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 80

#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.1 --loss 2 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60
#python train.py --lr 5e-4 --adp 0.1 --fdp 0.1 --encoders 5 --ln 2 --samp 0.01 --theta 0.1 --loss 3 --worope --gpu 1 --dataset AS12_lag3  --batch_size 128 --epoch 60

python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --encoders 4 --loss 1 --ln 1 --gpu 0 --wof --dataset AS09_lag --batch_size 64 --epoch 150
python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --encoders 4 --loss 2 --ln 1 --gpu 0 --wof --dataset AS09_lag --batch_size 64 --epoch 150
python train.py --lr 5e-4 --samp 0.01 --theta 0.1 --encoders 4 --loss 3 --ln 1 --gpu 0 --wof --dataset AS09_lag --batch_size 64 --epoch 150
