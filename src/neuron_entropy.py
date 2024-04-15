import torch
import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import entropy, norm
import matplotlib.pyplot as plt

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='Analysing neuron activations in hidden layers')

# ==> Logging Parameters
parser.add_argument('--dir', default='results/drl/CartPole-v1')
parser.add_argument('--regex', default=None)
parser.add_argument('--xlsx_name', default=None)

# ==> Env Parameters
parser.add_argument('--num_actions', default=1)
parser.add_argument('--n_eval', default=50)
parser.add_argument('--n_steps_max', default=10000)
parser.add_argument('--reward_threshold', default=195, type=int)

# ==> Plot Parameters
parser.add_argument('--plot_avg_entropy', default=0)
parser.add_argument('--plot_init_ent_to_final_step', default=0)
parser.add_argument('--initial_ent_min', default=800)
parser.add_argument('--initial_ent_max', default=830)
parser.add_argument('--ent_steps', default=10)

# =========================
# === SUPPORT FUNCTIONS ===
# =========================

def get_activation_and_actions_tensors(args):
    files = []
    run_dirs = [os.path.join(args.dir, directory) for directory in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, directory))]
    # Parse Directories
    for dir in run_dirs:
        paths = os.listdir(dir)
        # ordered_paths = [os.path.splitext(path)[0] for path in paths]
        ordered_paths = [os.path.splitext(path)[0] for path in paths if path.endswith('.pt') and "atv_and_act" in path]
        ordered_paths = [(path, int(path.split("_")[-1])) for path in ordered_paths]
        ordered_paths = sorted(ordered_paths, key=lambda x: x[1])
        # files.append([os.path.join(dir, file, "model.zip") for file, _ in ordered_paths])
        files.append([os.path.join(dir, file+".pt") for file, _ in ordered_paths])

    # Load Tensors and Stack
    atv_act_tensors = []
    for run in files:
        sub_atv_act_tensors = []
        for path in run:
            tensor = torch.load(path)
            tensor.requires_grad = True
            sub_atv_act_tensors.append(tensor)
        atv_act_tensors.append(torch.vstack(sub_atv_act_tensors).cpu())
    
    return atv_act_tensors

def load_sheets(args, dir, xlsx_name):
    # Excel data source
    source_file = dir + "\\" + xlsx_name

    # Load and Process Eval Reward Sheet
    df_rew = pd.read_excel(open(source_file, 'rb'), sheet_name='eval mean_reward')
    df_rew = df_rew.loc[:,~df_rew.columns.str.match("Unnamed")] # Drop unnamed columns
    df_rew = df_rew.drop(columns=['step'])                      # Drop step column
    df_rew[df_rew < args.reward_threshold] = 0                  # Set all values below threshold to 0

    df_rew_step = []
    for col in df_rew.columns:
        sol_idx = -1
        for idx, rew in df_rew[col].items():
            if float(rew) > 0:
                sol_idx = idx
                break
        df_rew_step.append(sol_idx)

    # # get first nonzero element in each row
    # df_rew_step = []
    # for col in df_rew.columns:
    #     df_rew_step.append(df_rew[col].gt(0).idxmax())

    return df_rew_step

def get_out_file_name(dir, out_xlsx):
    if out_xlsx == None:
        project_name = os.path.normpath(dir).split(os.path.sep)[-1]# remove extension
        out_xlsx = project_name + ".xlsx"
    return out_xlsx

# =====================
# === MAIN FUNCTION ===
# =====================

def main(args):
    atv_act_tensors = get_activation_and_actions_tensors(args)
    xlsx_name = get_out_file_name(args.dir, args.xlsx_name)
    df_rew_step = load_sheets(args, args.dir, xlsx_name)

    actions = [0,1]
    entropy_values = []
    with torch.no_grad():
        for run_tensors in atv_act_tensors:
            run_activations_tensor = run_tensors[:, :-args.num_actions]
            run_actions_tensor = run_tensors[:, -args.num_actions:]
            run_entropy = []
            for i in range(0, run_tensors.shape[0], args.n_eval):
                step_activations = run_activations_tensor[i:i+args.n_eval, :128]
                step_actions = run_actions_tensor[i:i+args.n_eval]
                np_step_actions = step_actions.cpu().detach().numpy().squeeze()
                step_entropy = 0
                for act in actions:
                    act_idx = torch.from_numpy(np.where(np_step_actions == act)[0])
                    act_activations = torch.index_select(step_activations, 0, act_idx)
                    for idx in range(act_activations.shape[1]):
                        step_entropy += entropy(norm.pdf(act_activations[:, idx].detach().numpy().squeeze()))
                run_entropy.append(step_entropy)
            entropy_values.append(run_entropy)
        
    if args.plot_avg_entropy == 1:
        arr = np.zeros([len(entropy_values),20])
        for i,j in enumerate(entropy_values):
            arr[i][0:len(j)] = j
        plt.plot(np.mean(arr, axis=0))
        plt.show()
    
    if args.plot_init_ent_to_final_step == 1:
        initial_entropy_values = [e[0] for e in entropy_values]
        final_step = []
        for step in df_rew_step:
            if step != -1:
                final_step.append((step+1)*args.n_eval)
            else:
                final_step.append(args.n_steps_max)
        plt.scatter(initial_entropy_values, final_step)
        plt.show()

    total_iters = 0
    total_success = 0 
    for step in df_rew_step:
        if step != -1:
            total_success += 1
            total_iters += (step+1)*args.n_eval
        else:
            total_iters += args.n_steps_max
    total_sample_efficiency = total_success / total_iters
    print("TSE:" , total_sample_efficiency)

    sample_efficiency = []
    ent_thresholds = np.linspace(args.initial_ent_min, args.initial_ent_max, args.ent_steps)
    initial_entropy_values = [e[0] for e in entropy_values]
    for ent_th in ent_thresholds:
        total_iters = 0
        total_success = 0
        for idx, step in enumerate(df_rew_step):
            if initial_entropy_values[idx] < ent_th:
                if step != -1:
                    total_success += 1
                    total_iters += (step+1)*args.n_eval
                else:
                    total_iters += args.n_steps_max
            else:
                total_iters += args.n_eval
        sample_efficiency.append(total_success / total_iters)
    print("SE:" , sample_efficiency)
    print("SE Boost:" , [se/total_sample_efficiency for se in sample_efficiency])

    plt.plot(ent_thresholds, sample_efficiency)
    plt.plot(ent_thresholds, [se/total_sample_efficiency for se in sample_efficiency])
    plt.show()



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    tensors = get_activation_and_actions_tensors(args)
    for t in tensors:
        print(t.grad)

