#!/bin/bash

# Generates csv files from the tensorboard logs

for ((i=0;i<=9;i++)); 
do 
   python tensorboard_to_csv.py --dir "C:\Users\George\OneDrive\Documents\University\Y4F\Thesis\thesis-rl-project-main\thesis-rl-project-main\src\results\drl\CartPole-v0\final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00"
   echo $i
done
