#!/bin/bash

for ((i=1;i<=9;i++)); 
do 
   python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed ${i}00 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00 --lr 2.50e-3 --activation_fn "tanh"
   echo $i
done