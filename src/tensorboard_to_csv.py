#!/usr/bin/env python3

import os
import pprint
import traceback
from pathlib import Path
from tqdm import tqdm
import re

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

parser = argparse.ArgumentParser(description='tensorboard_to_csv')
parser.add_argument('--dir', default="results/drl/CartPole-v1/test_21900", type=str)
parser.add_argument('--out_xlsx', default=None, type=str)
parser.add_argument('--regex', default=None, type=str)

def reformat_df(df, df_num):
    metric_values = df["metric"].unique()
    df_dict = {metric:[] for metric in metric_values}
    
    for metric in metric_values:
        # Get sub df associated with specific metric (reset index counting)
        sub_df_metric = df.loc[df["metric"] == metric].reset_index()

        # Convert value column to metric, and only keep 'metric' & step column
        metric_name = metric + "_" + str(df_num)
        sub_df_metric = sub_df_metric.rename(columns={"value": metric_name})
        sub_df_metric = sub_df_metric.drop(columns=["metric", "index"])
        df_dict[metric].append(sub_df_metric[["step", metric_name]])

    # Outer join all metric on 'step'
    for metric in metric_values:
        merged_df = df_dict[metric][0]
        for idx in range(1, len(df_dict[metric])):
            merged_df = merged_df.merge(df_dict[metric][idx], how='outer', on="step")
        df_dict[metric] = merged_df
    
    return df_dict

# Extraction function
def tflog2pandas(path: str, idx: int) -> pd.DataFrame:
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return reformat_df(runlog_data, idx)

def merge_log_dict(dict1, dict2):
    if bool(dict1) == False:
        return dict2
    elif bool(dict2) == False:
        return dict1
    
    ret_dict = {}
    for k in dict1:
        ret_dict[k] = dict1[k].merge(dict2[k], how='outer', on="step")
    return ret_dict

def many_logs2pandas_sheets_dict(event_paths):
    all_logs_dict = {}
    fails = []
    for idx in tqdm(range(len(event_paths))):
        log = tflog2pandas(event_paths[idx], idx)
        if log is not None:
            try:
                all_logs_dict = merge_log_dict(all_logs_dict, log)
            except:
                fails.append(idx)
    print("Failed  on: ", fails)
    return all_logs_dict

def get_out_file_name(dir, out_xlsx):
    if out_xlsx == None:
        project_name = os.path.normpath(dir).split(os.path.sep)[-1]# remove extension
        out_file = project_name + ".xlsx"
    return out_file

def to_valid_sheet_name(sheet_name):
    return sheet_name.replace("/", " ").replace("\\", " ")

def parse(out_xlsx, dir):
    pp = pprint.PrettyPrinter(indent=4)
    # Get all event* runs from logging_dir subdirectories
    event_paths = list(Path(dir).rglob("events*"))
    event_paths = [str(file) for file in event_paths]

    # Call & append
    if event_paths:
        pp.pprint("Found "+str(len(event_paths))+" tensorflow logs to process")
        all_logs = many_logs2pandas_sheets_dict(event_paths)

        print("==> Saving to xlsx file")
        out_xlsx = os.path.join(dir, get_out_file_name(dir, out_xlsx))

        print("==> Outfile: ", out_xlsx)
        with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
            for sheet_name in all_logs:
                print("Processing: ", sheet_name)
                all_logs[sheet_name].to_excel(writer, sheet_name=to_valid_sheet_name(sheet_name))
    else:
        print("No event paths have been found.")

def main(args):
    if args.regex != None:
        dirs = os.listdir(args.dir)
        selected_dirs = [dir for dir in dirs if re.search(args.regex, dir)]
        for dir in selected_dirs:
            # We don't allow different naming per regex result as of now
            # Use default xlsx naming
            parse(None, args.dir + "\\" + dir)
    else:
        parse(args.out_xlsx, args.dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
