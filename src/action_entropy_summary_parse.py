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

parser = argparse.ArgumentParser(description='Apply Pseudo Action Entropy Threshold on parsed csv file')

# ==> Data Source
parser.add_argument('--dir', default="C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\drl\\CartPole-v0", type=str)
parser.add_argument('--csv_name', default=None, type=str)
parser.add_argument('--nrows', default=None, type=int)
parser.add_argument('--regex', default=None, type=str)
parser.add_argument('--results_name', default=None, type=str)

def get_out_file_name(dir, out_csv):
    if out_csv == None:
        project_name = os.path.normpath(dir).split(os.path.sep)[-1]# remove extension
        out_csv = project_name + "_summary.csv"
    return out_csv

def main(args):
    # Parse multiply based on Regex
    if args.regex != None:
        dirs = os.listdir(args.dir)
        selected_dirs = [dir for dir in dirs if re.search(args.regex, dir)]
        total_df = None
        for idx, dir in enumerate(selected_dirs):
            try:
                lr = float(dir[-7:])
            except:
                lr = 2.5e-3
            try:
                arch = int(dir[14:17])
            except:
                arch = 64
            csv_name = get_out_file_name(dir, args.csv_name)
            sheet_dir = args.dir + "\\" + dir
            df = pd.read_csv(sheet_dir + "\\" + csv_name, header=None)
            df_lr_arch = pd.DataFrame(data = [["LR", lr],["Num neurons", arch]], columns=[0,1])
            df = pd.concat([df, df_lr_arch])
            if not isinstance(total_df, pd.DataFrame):
                total_df = df
            else:
                total_df[idx+1] = df[1]
        total_df = total_df.set_index(0)
        total_df = total_df.sort_values(by=["Num neurons", "LR"], axis=1)
        results_name = args.results_name if args.results_name != None else "summary.csv"
        total_df.to_csv(results_name)
    # Parse single given dir
    else:
        raise NotImplementedError

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

