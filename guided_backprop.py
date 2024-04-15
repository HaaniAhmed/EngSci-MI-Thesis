from tkinter import N
import torch
import argparse
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import entropy, norm
import seaborn as sb
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3_thesis.common.torch_layers import MlpExtractor
from captum.attr import LayerGradCam

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='Analysing neuron activations\
     in hidden layers')

# ==> Logging Parameters
parser.add_argument('--dir', default='results\\drl\\Acrobot-v1\\test_10000')
parser.add_argument('--xlsx_name', default=None)
parser.add_argument('--heatmap_name', default="heatmap.png")
parser.add_argument('--heatmap_percent_name', default="heatmap_pc.png")

# ==> Env Parameters
parser.add_argument('--num_actions', default=1)
parser.add_argument('--n_eval', default=50)
parser.add_argument('--n_env_eval', default=500)
parser.add_argument('--n_steps_max', default=50000)
parser.add_argument('--reward_threshold', default=-100, type=int)

# ==> Threshold Parameters
parser.add_argument('--task', default='neuron_entropy_grad')
parser.add_argument('--threshold_steps', default=11, type=int)
parser.add_argument('--grad_threshold_steps', default=4, type=int)
parser.add_argument('--weight_grad', default=0, type=int, help='whether to weight\
     neuron activation entropy by grad magnitude')


def get_tensors_and_model(args):
    files_actvs = []
    files_feats = []
    files_model = []

    run_dirs = [os.path.join(args.dir, directory) for directory in \
        os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, directory))]
    run_dirs = [(path, int(path.split("_")[-1])) for path in run_dirs]
    run_dirs = sorted(run_dirs, key=lambda x: x[1])
    run_dirs = [dir for dir, _ in run_dirs]

    # Parse Directories
    for dir in run_dirs:
        paths = os.listdir(dir)

        ordered_paths_actvs = [os.path.splitext(path)[0] for path in paths\
             if path.endswith('.pt') and path.startswith('atv_and_act')]
        ordered_paths_actvs = [(path, int(path.split("_")[-1])) for path in\
             ordered_paths_actvs]
        ordered_paths_actvs = sorted(ordered_paths_actvs, key=lambda x: x[1])

        ordered_paths_feats = [os.path.splitext(path)[0] for path in paths\
             if path.endswith('.pt') and path.startswith('features')]
        ordered_paths_feats = [(path, int(path.split("_")[-1])) for path in\
             ordered_paths_feats]
        ordered_paths_feats = sorted(ordered_paths_feats, key=lambda x: x[1])

        ordered_paths_model = [os.path.splitext(path)[0] for path in paths\
             if path.endswith('.pt') and path.startswith('model')]
        ordered_paths_model = [(path, int(path.split("_")[-1])) for path\
             in ordered_paths_model]
        ordered_paths_model = sorted(ordered_paths_model, key=lambda x: x[1])

        files_actvs.append([os.path.join(dir, file+".pt")\
            for file, _ in ordered_paths_actvs])
        files_feats.append([os.path.join(dir, file+".pt")\
            for file, _ in ordered_paths_feats])
        files_model.append([os.path.join(dir, file+".pt")\
            for file, _ in ordered_paths_model])

        with open('model.json', 'w') as f:
            json.dump(files_model, f)

    # Load Activation Tensors and Stack
    atv_tensors = []
    act_tensors = []
    feats_tensors = []
    models = []

    for actvs_run, feats_run, model_run in zip(files_actvs, \
        files_feats, files_model):
        sub_atv_tensors = []
        sub_act_tensors = []
        sub_feats_tensors = []
        sub_models = []
        for actvs_path, feats_path, model_path in \
            zip(actvs_run, feats_run, model_run):
            atv_act_tensor = torch.load(actvs_path, map_location=torch.device('cpu')).cpu()

            # DONE: change this bc it is currently hardcoded to old cartpole model
            sub_atv_tensors.append(atv_act_tensor[:, :32])  # NEW: was 128 before
            sub_act_tensors.append(atv_act_tensor[:, 32:])
            sub_feats_tensors.append(torch.load(feats_path, map_location=torch.device('cpu')).cpu())
            model = MlpExtractor(feature_dim=6, \
                net_arch=[dict(pi=[16, 16], vf=[16, 16])], activation_fn=nn.ReLU).cpu() # NEW: was 64 before, and tanh
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.gradcam_forward = True
            sub_models.append(model)

        act_tensors.append(torch.vstack(sub_act_tensors))
        atv_tensors.append(torch.vstack(sub_atv_tensors))
        feats_tensors.append(torch.vstack(sub_feats_tensors))
        models.append(sub_models)
    print(models)
    print(atv_tensors)
    print(act_tensors)
    print(feats_tensors)
    return models, atv_tensors, act_tensors, feats_tensors

def get_out_file_name(dir, out_xlsx):
    if out_xlsx == None:
        project_name = os.path.normpath(dir).split(os.path.sep)[-1]# remove extension
        out_xlsx = project_name + ".xlsx"
    return out_xlsx

def load_sheets(args, dir, xlsx_name):
    # Excel data source
    source_file = dir + "\\" + xlsx_name

    # Load and Process Eval Reward Sheet
    df_rew = pd.read_excel(open(source_file, 'rb'), sheet_name='eval mean_reward')
    df_rew = df_rew.loc[:,~df_rew.columns.str.match("Unnamed")] # Drop unnamed columns
    df_rew = df_rew.drop(columns=['step'])                      # Drop step column
    df_rew[df_rew < args.reward_threshold] = 0

    df_rew_step = []
    for col in df_rew.columns:
        sol_idx = -1
        for idx, rew in df_rew[col].items():
            if float(rew) > 0:
                sol_idx = idx
                break
        df_rew_step.append((sol_idx+1)*args.n_env_eval if sol_idx != -1 else sol_idx)
    return df_rew_step

def action_entropy(args, actions, rew_step, threshold):
    sample_efficiency = 0
    num_success = 0
    num_iters = 0
    for i in range(len(rew_step)):
        action_entropy = entropy
    return sample_efficiency

def get_neuron_activation_with_grads(feats, model, actions, activations):
    # Calculate neuron entropy per run
    # Neuron activation entropy calculated w.r.t. each action, for each neuron
    action_values = [0,1]
    atv_with_grad = [] # [run1, run2, ...] -> run1={action1:((atv1, grad1), ...
    for f, m, a, av in zip(feats, model, actions, activations):
        action_run_atv_with_grad = {}
        for act in action_values:
            act_idx = torch.from_numpy(np.where(a == act)[0])
            act_activations = av[act_idx, :]
            sub_run_atv_with_grad = []
            for idx in range(act_activations.shape[1]):
                neuron_activation = act_activations[:, idx].detach().numpy().squeeze()
                layer_gradcam = LayerGradCam(m, m.policy_net[0] if idx < 16  else m.policy_net[2]) # NEW
                grad = layer_gradcam.attribute(f, target=idx% 16).squeeze().detach().numpy()
                sub_run_atv_with_grad.append((neuron_activation, grad))
            action_run_atv_with_grad[act] = sub_run_atv_with_grad
        atv_with_grad.append(action_run_atv_with_grad)

    return atv_with_grad

def get_neuron_entropy(actions, activations):
    # Calculate neuron entropy per run
    # Neuron activation entropy calculated w.r.t. each action, for each neuron
    action_values = [0, 1, 2] # TODO: hardcoded!
    run_entropy = []
    for a, av in zip(actions, activations):
        neuron_entropy = 0
        for act in action_values:
            act_idx = torch.from_numpy(np.where(a == act)[0])
            act_activations = av[act_idx, :]
            for idx in range(act_activations.shape[1]):
                neuron_entropy += entropy(norm.pdf(\
                    act_activations[:, idx].detach().numpy().squeeze()))
        run_entropy.append(neuron_entropy)
    return run_entropy

def neuron_entropy(args, actions, activations, rew_step, threshold):
    # Calculate the sample efficiency boost for some threshold percent
    run_entropy = get_neuron_entropy(actions, activations)
    threshold_idx = int(threshold*len(run_entropy)) if threshold != 1 \
        else len(run_entropy)-1
    threshold_val = sorted(run_entropy)[threshold_idx]
    total_iters = 0
    total_success = 0
    for idx, step in enumerate(rew_step):
        if run_entropy[idx] < threshold_val:
            if step != -1: # TODO: this might be hardcoded
                total_success += 1
                total_iters += step
            else:
                total_iters += args.n_steps_max
        else:
            total_iters += args.n_eval
    sample_efficiency = total_success / total_iters
    return sample_efficiency

def neuron_entropy_with_grad(args, run_activations, run_entropy, rew_step,\
     neuron_threshold, grad_threshold, weight_grad=0):
    # Calculate Neuron Entropy
    run_with_grad_entropy = []
    for run in run_activations:
        run_ent_val = 0
        for _, act_and_grad in run.items():
            activation = torch.tensor([i[0] for i in act_and_grad])
            gradients = torch.tensor([i[1] for i in act_and_grad])
            grad_mean = torch.abs(torch.mean(gradients, axis=1))
            k = min(int(grad_threshold*32), 31)  # NEW
            grad_mags, topk_idxs = torch.topk(grad_mean, k)
            if len(topk_idxs) != 0:
                for grad_mag, idx in zip(grad_mags, topk_idxs):
                    if weight_grad == 0:
                        run_ent_val += entropy(norm.pdf(\
                            activation[idx, :].detach().numpy().squeeze()))
                    elif weight_grad == 1:
                        run_ent_val += grad_mag.item()*entropy(\
                            norm.pdf(activation[idx, :].detach().numpy()\
                                .squeeze()))
        run_with_grad_entropy.append(run_ent_val)

    # Calculate the sample efficiency boost for some threshold percent
    threshold_idx = int(neuron_threshold*len(run_entropy)) if \
        neuron_threshold != 1 else len(run_entropy)-1
    threshold_val = sorted(run_with_grad_entropy)[threshold_idx]
    total_iters = 0
    total_success = 0
    for idx, step in enumerate(rew_step):
        if run_with_grad_entropy[idx] < threshold_val:
            if step != -1:
                total_success += 1
                total_iters += step
            else:
                total_iters += args.n_steps_max
        else:
            total_iters += args.n_eval
    sample_efficiency = total_success / total_iters
    return sample_efficiency

def main(args):
    models, activations, actions, feats = get_tensors_and_model(args)
    xlsx_name = get_out_file_name(args.dir, args.xlsx_name)
    rew_step = load_sheets(args, args.dir, xlsx_name)

    first_models = [m[1] for m in models]
    first_feats = [f[50:args.n_eval+50, :] for f in feats]
    first_actions = [a[50:args.n_eval+50, :] for a in actions]
    first_activations = [a[50:args.n_eval+50, :] for a in activations]

    if args.task == "neuron_entropy":
        ne_boosts = []
        thresholds = np.linspace(0, 1, args.threshold_steps)
        for threshold in thresholds:
            ne = neuron_entropy(args, first_actions, first_activations, \
                rew_step, threshold)
            ne_boosts.append(ne)

        plt.plot(thresholds, ne_boosts)
        plt.show()
    elif args.task == "neuron_entropy_grad":
        atv_with_grad = get_neuron_activation_with_grads(first_feats, 
                                                        first_models, 
                                                        first_actions, 
                                                        first_activations)
        run_entropy = get_neuron_entropy(first_actions, first_activations)

        thresholds = np.around(np.linspace(0, 1, args.threshold_steps)\
            , decimals=1)
        grad_thresholds = np.around(np.linspace(\
            0, 1, args.grad_threshold_steps), decimals=1)
        se_boosts = np.zeros((args.grad_threshold_steps, args.threshold_steps))
        for n_idx, n_threshold in enumerate(thresholds):
            for ng_idx, ng_threshold in enumerate(grad_thresholds):
                se_boosts[ng_idx, n_idx] = neuron_entropy_with_grad(args, \
                    atv_with_grad, run_entropy, rew_step, n_threshold, \
                    ng_threshold, args.weight_grad)
                print("PROGRESS: ", n_idx, ng_idx)
        
        sb.heatmap(se_boosts, xticklabels=thresholds,\
             yticklabels=grad_thresholds)
        plt.tight_layout()
        plt.savefig(args.heatmap_name)

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        pc_boosts = se_boosts/max(se_boosts[0, 0], se_boosts[0, -1], \
            se_boosts[-1, 0], se_boosts[-1,-1])
        print("MAX BOOST: ", np.amax(pc_boosts))
        print("VAL BOOST 0.7: ", pc_boosts[-1, int(pc_boosts.shape[1]*0.7)])
        print("VAL BOOST 0.8: ", pc_boosts[-1, int(pc_boosts.shape[1]*0.8)])

        sb.heatmap(pc_boosts, xticklabels=thresholds, yticklabels=grad_thresholds)
        plt.tight_layout()
        plt.savefig(args.heatmap_percent_name)

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

