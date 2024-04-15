#!/bin/bash

python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 16
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 32
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 48
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 64
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 80
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 96
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 112
python srl.py --max_iters 100000 --dataset_size 1000 --batch_size 25 --num_neurons_per_layer 128

# python action_entropy.py --base_folder "C:\Users\Michael Ruan\Documents\Github\thesis-rl-project\src\results\drl\CartPole-v0" --regex "^srl_n_env_10_arch_016_016*" --srl_steps_offset 10000
# python action_entropy.py --base_folder "C:\Users\Michael Ruan\Documents\Github\thesis-rl-project\src\results\drl\CartPole-v0" --regex "^srl_n_env_10_arch_128_128*" --srl_steps_offset 10000
# python tensorboard_to_csv.py --dir "C:\Users\Michael Ruan\Documents\Github\thesis-rl-project\src\results\drl\CartPole-v0" --regex "^srl_1000samples_n_env_10_arch_016_016*"
# python action_entropy.py --base_folder "C:\Users\Michael Ruan\Documents\Github\thesis-rl-project\src\results\drl\CartPole-v0" --regex "^srl_1000samples_n_env_10_arch_016_016*" --srl_steps_offset 1000