import os
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm import tqdm

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='Apply Pseudo Action \
    Entropy Threshold on parsed xlsx file')

# ==> Data Source
parser.add_argument('--dir', default="results\\drl\\Acrobot-v1", type=str)
parser.add_argument('--xlsx_name', default=None, type=str)
parser.add_argument('--nrows', default=None, type=int)
parser.add_argument('--regex', default=None, type=str)

# ==> Env Solution (If not NaN in sheet)
parser.add_argument('--reward_threshold', default=-100, type=int)

# ==> Entropy Thresholding Arguments
parser.add_argument('--srl_steps_offset', default=0, type=int)
parser.add_argument('--compute_boosts', default=1, type=int)
parser.add_argument('--plot_heatmap', default=0, type=int)
parser.add_argument('--drop_failures', default=False, type=bool)
parser.add_argument('--et_steps', default=50, type=int)
parser.add_argument('--et_steps_max_diff', default=0.5, type=float)
parser.add_argument('--steps_max', default=200, type=int)
parser.add_argument('--steps_freq', default=50, type=int)
parser.add_argument('--steps_offset', default=50, type=int)

# ==> Visualization
parser.add_argument('--plot_title', default=None, type=str)

# ==> Logging
parser.add_argument('--plot_filename', default=None, type=str)
parser.add_argument('--boosts_filename', default=None, type=str)


# =========================
# === SUPPORT FUNCTIONS ===
# =========================

def load_sheets(args, dir, xlsx_name):
    # Excel data source
    source_file = dir + "\\" + xlsx_name

    # Load and Process Entropy Sheet
    try:
        df_ent = pd.read_excel(open(source_file, 'rb'), \
            sheet_name='train action_entropy', nrows=args.nrows)
    except:
        df_ent = pd.read_excel(open(source_file, 'rb'), \
            sheet_name='train entropy_loss', nrows=args.nrows)
    df_ent = df_ent.loc[:,~df_ent.columns.str.match("Unnamed")] # Drop unnamed columns
    df_ent = df_ent.set_index('step')   # Set step to index
    df_ent = np.exp(df_ent)             # Apply e^x to sheet
    df_ent = df_ent - 0.5               # Get distance to 0.5
    df_ent[df_ent < 0] = 0              # Make negative numbers 0

    # Load and Process Eval Reward Sheet
    df_rew = pd.read_excel(open(source_file, 'rb'), sheet_name='eval mean_reward')
    df_rew = df_rew.loc[:,~df_rew.columns.str.match("Unnamed")] # Drop unnamed columns
    df_rew = df_rew.drop(columns=['step'])                      # Drop step column
    df_rew[df_rew < args.reward_threshold] = np.nan

    # Total Number of Runs
    num_runs = len(df_ent.columns)

    # Get Failed Runs
    rew_cols_to_drop = []
    ent_cols_to_drop = []
    for ent_col, rew_col in zip(df_ent.columns, df_rew.columns):
        if df_rew[rew_col].isnull().values.all() == True:
            rew_cols_to_drop.append(rew_col)
            ent_cols_to_drop.append(ent_col)

    # Number of Failed Runs
    num_fail = len(rew_cols_to_drop)

    # Drop Failed Runs if Needed
    if args.drop_failures == True:
        df_rew = df_rew.drop(columns=rew_cols_to_drop)
        df_ent = df_ent.drop(columns=ent_cols_to_drop)
        print("Dropping ", len(rew_cols_to_drop), " due to failures")

    df_rew_bool = df_rew.notna().any(axis=0)
    return df_ent, df_rew_bool, num_runs, num_fail

def get_out_file_name(dir, out_xlsx):
    if out_xlsx == None:
        project_name = os.path.normpath(dir).split(os.path.sep)[-1]# remove extension
        out_xlsx = project_name + ".xlsx"
    return out_xlsx

def get_total_sample_efficiency(df_ent, df_rew):
    total_iters = 0
    total_success = sum(list(df_rew))
    for col in df_ent.columns:
        last_val_idx = df_ent[col].last_valid_index()
        total_iters += last_val_idx
    return total_iters, total_success

def apply_entropy_threshold(df, ent_step, ent_val, df_rew, srl_steps_offset):
    thresholded_iters = 0
    thresholded_success = 0
    rew_list = list(df_rew)

    # Iterate columns (i.e. iterate all runs)
    for idx, col in enumerate(df.columns):
        thresholded_iters += srl_steps_offset
        col_ent_val_at_step = df[col].loc[ent_step]
        if col_ent_val_at_step > ent_val:
            thresholded_iters += df[col].last_valid_index()
            thresholded_success += rew_list[idx]
        else:
            thresholded_iters += ent_step

    sample_efficiency = thresholded_success/thresholded_iters
    return sample_efficiency

def set_tick_visibility(ticklabels, freq, disp_last=True):
    for ind, label in enumerate(ticklabels):
        if ind % freq == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    if disp_last == True:
        ticklabels[-1].set_visible(True)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

# =====================
# === MAIN FUNCTION ===
# =====================

def pseudo_action_entropy(args, dir, xlsx_name):
    '''Finds the ideal action entropy threshold value/step
    to get maximum sample efficiency improvement'''
    # Excel source data filename (no ext)
    source = os.path.splitext(xlsx_name)[0]

    # Get action entropy (differences to 0.5 action entropy) and rewards
    df_ent, df_rew, num_runs, num_fail = load_sheets(args, dir, xlsx_name)

    # Get steps in list
    steps = df_ent.index.tolist()
    steps = [x for x in steps if x <= args.steps_max]
    steps.sort()

    # Get entropy thresholds to test
    df_ent_list = df_ent.iloc[[1]].values.flatten().tolist()
    df_ent_list.sort()
    #thresholds = list(np.linspace(et_min, et_max, num=args.et_steps+1))
    thresholds = df_ent_list[::2]
    thresholds = [round(num, 4) for num in thresholds] # Round to 3 decimal places
    #print("TH: ", thresholds)

    # Get max entropy value
    sample_efficiency_boosts = np.zeros((len(steps), len(thresholds)))
    total_iters, total_success = get_total_sample_efficiency(df_ent, df_rew)
    total_iters += args.srl_steps_offset * num_runs
    total_sample_efficiency = total_success/total_iters
    print("TOTAL SAMPLE EFFIECIENCY: ", total_sample_efficiency)

    # Whether to compute the heatmap
    if args.compute_boosts == 1:
        for step_idx, step in tqdm(enumerate(steps)):
            for threshold_idx, threshold in enumerate(thresholds):
                thresholded_sample_efficiency = apply_entropy_threshold(df_ent, \
                    step, threshold, df_rew, args.srl_steps_offset)
                sample_efficiency_boosts[step_idx, threshold_idx] = \
                    (thresholded_sample_efficiency - total_sample_efficiency)\
                    /total_sample_efficiency

        boosts_filename = args.boosts_filename if args.boosts_filename != None \
            else dir + "\\" + source + "_boosts.csv"
        np.savetxt(boosts_filename, sample_efficiency_boosts, delimiter=",")

        if args.plot_heatmap == 1:
            # Plot heat map
            steps_str = [str(x + args.srl_steps_offset) for x in steps]
            thresholds_str = [str(x) for x in thresholds]
            heatmap = sb.heatmap(sample_efficiency_boosts, \
                xticklabels=thresholds_str, yticklabels=steps_str)
            
            # Adjust axes tick frequency
            set_tick_visibility(heatmap.get_xticklabels(), \
                freq=int(len(thresholds_str)/10), disp_last=True)
            set_tick_visibility(heatmap.get_yticklabels(), \
                freq=int(len(steps_str)/10), disp_last=True)

            # Set axis/plot title
            plot_title = args.plot_title if args.plot_title\
                != None else source
            plot_filename = args.plot_filename if args.plot_filename \
                != None else dir + "\\" + source + ".png"
            plt.title(plot_title)
            plt.xlabel('thresholds (diff to 0.5)')
            plt.ylabel('steps')
            plt.tight_layout()
            plt.savefig(plot_filename)

            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()

    # To computer later parts, we need sample efficiency boosts. 
    # In this case, we assume the boosts were calculated before
    # with the right naming
    elif args.compute_heatmap == 0:
        boosts_filename = args.boosts_filename if args.boosts_filename\
            != None else dir + "\\" + source + "_boosts.csv"
        sample_efficiency_boosts = pd.read_csv(boosts_filename, \
            sep=',',header=None).to_numpy()

    # Save Sample Efficiency & Best ET Parameters Summary
    log_step = args.steps_freq
    log_step_offset = args.steps_offset

    log_threshold = args.et_steps_max_diff / args.et_steps
    log_threshold_offset = 0.5 - args.et_steps_max_diff

    max_et_loc = np.unravel_index(sample_efficiency_boosts.argmax(), \
        sample_efficiency_boosts.shape)
    ent_val_list = df_ent.iloc[[1]].values.flatten().tolist()
    ent_val_list.sort()
    ent_val = ent_val_list[int(len(ent_val_list)*0.75)]
    _, ent_idx = find_nearest(thresholds, ent_val)
    max_et_loc_conv = (1, ent_idx)
    print(max_et_loc_conv)

    data = [
        ["Total Runs", num_runs],
        ["Total Fails", num_fail],
        ["No ET Success Rate: ", (num_runs-num_fail)/num_runs],
        ["Total Sample Efficiency", total_sample_efficiency],
        ["Sample Efficiency Boost", \
            sample_efficiency_boosts[max_et_loc_conv[0], \
            max_et_loc_conv[1]]],
        ["Max Sample Efficiency Boost", \
            np.amax(sample_efficiency_boosts)],
        ["Sample Efficiency ", \
            (1+sample_efficiency_boosts[max_et_loc_conv[0], \
            max_et_loc_conv[1]])*total_sample_efficiency],
        ["Best ET Value (step)", \
            (1*log_step)+log_step_offset+args.srl_steps_offset],
        ["Best ET Value (action entropy)", max_et_loc_conv[1]],
        ["SRL Iters", args.srl_steps_offset]
    ]
    df_log = pd.DataFrame(data)
    df_log.to_csv(dir + "\\" + source + "_summary.csv", \
        index=False, header=False)

def action_entropy_to_steps(args):
    # Get action entropy (differences to 0.5 action entropy) and rewards dataframe
    xlsx_name = get_out_file_name(args.dir, args.xlsx_name)
    df_ent, df_rew, _, _ = load_sheets(args, args.dir, xlsx_name)

    act_ent_vals = []
    step_vals = []
    max_steps = df_ent.index[-1]
    for col in df_ent.columns:
        initial_act_ent = df_ent[col].iloc[0]
        solved_idx = df_ent[col].last_valid_index()
        if solved_idx == None:
            solved_idx = max_steps
        act_ent_vals.append(initial_act_ent)
        step_vals.append(solved_idx)
    

    plt.scatter(act_ent_vals, step_vals)
    plt.xlim(0, 2e-6)
    plt.ylim(0, 100000)
    plt.xlabel("First step's action entropy")
    plt.ylabel("Solved step")
    plt.show()
        

def main(args):
    # Parse multiply based on Regex
    if args.regex != None:
        dirs = os.listdir(args.dir)
        selected_dirs = [dir for dir in dirs if re.search(args.regex, dir)]
        for dir in selected_dirs:
            xlsx_name = get_out_file_name(dir, args.xlsx_name)
            pseudo_action_entropy(args, \
                args.dir + "\\" + dir, get_out_file_name(dir, xlsx_name))
    # Parse single given dir
    else:
        xlsx_name = get_out_file_name(args.dir, args.xlsx_name)
        pseudo_action_entropy(args, args.dir, xlsx_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    #action_entropy_to_steps(args)
