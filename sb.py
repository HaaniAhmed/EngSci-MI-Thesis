import os
from stable_baselines3_thesis.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3_thesis.common.env_util import make_vec_env
from stable_baselines3_thesis import A2C, DQN, DDPG, PPO, TD3
import argparse
import pickle
import gym
import torch
import random
import numpy as np
from tqdm import tqdm

from common import *

# New
import time

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='DRL training w/ Stable Baselines3')

# ==> DRL Parameters
parser.add_argument('--model', default='A2C', choices=models.keys())
parser.add_argument('--policy', default='MlpPolicy', choices=policies)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--lr_schedule', default="linear", type=str)
parser.add_argument('--num_neurons_per_layer', default=64, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--activation_fn', default="relu", type=str)

# ==> Env Parameters
parser.add_argument('--env', default='LunarLander-v2')
parser.add_argument('--steps', default=125000, type=int)
parser.add_argument('--n_env', default=1, type=int)
parser.add_argument('--disable_reward_threshold', default=0, type=int)
parser.add_argument('--reward_threshold', default=195, type=int)
parser.add_argument('--reward_threshold_episodes', default=100, type=int)
parser.add_argument('--eval_freq', default=1000, type=int)  # eval_freq --> eval_freq * n_env

# ==> DRL Framework Specific Parameters
parser.add_argument('--n_steps', default=5, type=int, help="number of updates before applying gradient (A2C, PPO)")
parser.add_argument('--ent_coef', default=0.02, type=float)
# parser.add_argument('--epsilon_initial', default=1.0, type=float)
# parser.add_argument('--epsilon_final', default=0.1, type=float)
# parser.add_argument('--target_update_interval', default=10000, type=int)
# parser.add_argument('--buffer_size', default=50000, type=int)

# ==> Logging Parameters
parser.add_argument('--project', '-pj', default="test")
parser.add_argument('--repeat', '-r', default=100, type=int)

# ==> Other Parameters
parser.add_argument('--model_path', default=None, help='model to load')
parser.add_argument('--weights_path', default=None, help='weights to load (pkl dictionary)')
parser.add_argument('--seed', default=1, type=int, help='random seed (-1=Random Seed)')
parser.add_argument('--device', default='cpu', help='device')
parser.add_argument('--arch', default='64_128_64', help='neurons in each hidden layer')

# =========================
# === SUPPORT FUNCTIONS ===
# =========================

def make_env(args):
    if args.model == "A2C" or args.model == "PPO":
        env = make_vec_env(args.env, n_envs=args.n_env)
    else: # Cannot have multiple environments
        env = gym.make(args.env)
    return env

def make_model(args, env, i):
    tfboard_log_folder = "./results/drl/"+args.env+"/"+args.project + "_" + str(args.seed) + "/"
    
    # Load only weights
    if args.weights_path != None:
        with open(args.weights_path, 'rb') as handle:
            drl_new_params = pickle.load(handle)
        model = get_model(args, env, tfboard_log_folder, i)
        model.set_parameters(drl_new_params, exact_match=False)

    # Load whole model
    elif args.model_path != None:
        if args.model in models:
            model = models.load(args.model_path, env)
        else:
            raise NotImplementedError
    
    # Create new model
    else:
        model = get_model(args, env, tfboard_log_folder, i)
    return model, tfboard_log_folder

# =====================
# === MAIN FUNCTION ===
# =====================


def get_entropy(activations):
    hist_activations = np.histogram(activations.cpu().numpy().squeeze(), bins=7)
    hist_counts = hist_activations[0] + 0.000001 # add epsilon to avoid log(0)
    hist_freqs = hist_counts / np.sum(hist_counts)
    run_ent_val = -np.sum(hist_freqs * np.log(hist_freqs, out=np.zeros_like(hist_freqs)))
    return run_ent_val


def drl(args, i):
    env = make_env(args)
    if args.disable_reward_threshold == 0:
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold)
        eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, eval_freq=args.eval_freq, n_eval_episodes=args.reward_threshold_episodes)
        if args.device == 'gpu':
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, tf_log = make_model(args, env, i)
        print("running on ", args.device)

        #model = model.to(device)
        # ADDED CODE
        # Save activations
        act_folder = "activations"
        if not os.path.isdir(act_folder):
            os.makedirs(act_folder)
        # create file in act_folder for i-th run
        acts_csv = open(act_folder + '/' + args.model + "_" + str(args.seed + i) + ".csv", "w")

        def get_activation(name):
            def hook(model, input, output):
                ent_val = get_entropy(output.detach())
                if name == 1:
                    acts_csv.write(str(ent_val) + ",")
                elif name == 3:
                    acts_csv.write(str(ent_val) + "\n")
            return hook
        model.policy.mlp_extractor.policy_net[1].register_forward_hook(get_activation(1))
        model.policy.mlp_extractor.policy_net[3].register_forward_hook(get_activation(3))
        # END ADDED CODE

        model.learn(total_timesteps=args.steps, eval_freq=1, n_eval_episodes=1, log_interval=1, callback=eval_callback)

        # acts_csv.close()

        model.save(tf_log + args.model + "_" + str(i + 1) + "/model")
    else:
        model = make_model(args, env)
        model.learn(total_timesteps=args.steps, eval_freq=1, n_eval_episodes=1, log_interval=1)
        model.save(tf_log + "model" + str(i+1))


def main(args):
    # TO DO: Parallelize environments
    all_times = []
    for i in range(args.repeat):
        print("\n\nstarting seed ", args.seed + i, "\n\n")
        start = time.perf_counter()
        torch.manual_seed(args.seed + i)
        np.random.seed(args.seed + i)
        random.seed(args.seed + i)
        drl(args, i)
        end = time.perf_counter()
        all_times.append(end - start)
        print(f"FINISHED IN {end - start} s")
    print(f"AVG TIME: {sum(all_times)/len(all_times)}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
