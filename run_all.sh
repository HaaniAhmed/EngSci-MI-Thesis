#!/bin/bash

for i in {93..99}; do
    echo "Running seed ${i}00"
  (
    python sb.py --model A2C --policy MlpPolicy --repeat 100 --seed ${i}00 --project final_n_env_10_arch_064_064_lr_2.50e-3_seed_0${i}00 --lr 2.50e-3 --activation_fn "tanh"
  ) & # Run the command in the background for parallel execution
  # wait a second
    sleep 1
done
wait # Wait for all background processes to finish before exiting the script