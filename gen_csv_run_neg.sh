#!/bin/bash

# Generates csv files from the tensorboard logs and runs the neuron entropy grad script on them

for ((i=18;i<=99;i++)); 
do 
    (
      python tensorboard_to_csv.py --dir "C:\Users\George\OneDrive\Documents\University\Y4F\Thesis\thesis-rl-project-main\thesis-rl-project-main\src\results\drl\CartPole-v0\final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00"   
    ) & # Run the command in the background for parallel execution 
   # python neuron_entropy_grad.py --dir "C:\Users\George\OneDrive\Documents\University\Y4F\Thesis\thesis-rl-project-main\thesis-rl-project-main\src\results-tanh\drl\CartPole-v0\final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00" \
   #  --xlsx_name "final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00.xlsx" --act_fcn "tanh" --method "gradcam"
   echo $i
done