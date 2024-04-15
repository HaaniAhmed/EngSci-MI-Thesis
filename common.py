from stable_baselines3_thesis import A2C, DQN, DDPG, PPO, TD3
from typing import Callable
import torch.nn as nn

models = {"A2C":A2C, "DQN":DQN, "DDPG":DDPG, "PPO":PPO, "TD3":TD3}
policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule"""
    def func(progress_remaining: float) -> float:
        """Progress will decrease from 1 (beginning) to 0"""
        return progress_remaining * initial_value
    return func

def get_model(args, env, tfboard_log_folder, i):
    # Get Learning Rate Schedule (if any)
    if args.lr_schedule == None:
        lr = args.lr
    elif args.lr_schedule == "linear":
        lr = linear_schedule(args.lr)

    # Creates a new model
    if args.model == "A2C":
        # Specify network parameters: actor=pi, value function=vf
        # print(args.activation_fn)
        if args.activation_fn == "relu":
            act_fn = nn.ReLU
        elif args.activation_fn == "tanh":
            act_fn = nn.Tanh
        # print(act_fn)
        neuron_list = [args.num_neurons_per_layer for _ in range(args.num_layers)]
        if args.arch == "64_128_64":
            neuron_list = [64, 128, 64]
        print("neuron list: ", neuron_list)
        policy_kwargs = dict(net_arch=[dict(pi=neuron_list, vf=neuron_list)],
                             activation_logdir=tfboard_log_folder,
                             activation_fn=act_fn)

        model = A2C(args.policy, env, learning_rate=lr, n_steps=args.n_steps, ent_coef=args.ent_coef, policy_kwargs=policy_kwargs,
                    tensorboard_log=tfboard_log_folder, seed=args.seed + i, device=args.device)
        # activations = {}
        # def get_activation(name):
        #     def hook(model, input, output):
        #         activations[name] = output.detach()
        #     return hook
        # model.policy.mlp_extractor.policy_net[1].register_forward_hook(get_activation(1))
        # model.policy.mlp_extractor.policy_net[3].register_forward_hook(get_activation(3))
        # print(activations)
        # #mlp_extractor.policy_net
    elif args.model == "DQN":
        model = DQN(args.policy, env, learning_rate=lr, # buffer_size=args.buffer_size,
                    # target_update_interval=args.target_update_interval, exploration_initial_eps=args.epsilon_initial, 
                    # exploration_final_eps=args.epsilon_final, 
                    tensorboard_log=tfboard_log_folder)
    elif args.model == "DDPG":
        model = DDPG(args.policy, env, learning_rate=lr, buffer_size=args.buffer_size,
                    tensorboard_log=tfboard_log_folder)
    elif args.model == "PPO":
        model = PPO(args.policy, env, learning_rate=lr, ent_coef=args.ent_coef, tensorboard_log=tfboard_log_folder)
    elif args.model == "TD3":
        model = TD3(args.policy, env, learning_rate=lr, buffer_size=args.buffer_size,
                    tensorboard_log=tfboard_log_folder)
    else:
        raise NotImplementedError
    return model
