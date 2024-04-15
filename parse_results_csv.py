import os
import csv

# create csv file to store results
results = open('experiment_summary.csv', 'w', newline='')
# write header using csv writer
csv_writer = csv.writer(results)
csv_writer.writerow(['file', 'type', 'activation', 'threshold_steps', 'env', 'control_se', 'highest_se', 'highest_se_boost (%)', 'highest_se_threshold'])

train_rows = []
test_rows = []


# loop through all csv files in current directory
for file in os.listdir(os.getcwd()):
    if file.endswith(".csv") and file != 'experiment_summary.csv' and "gradcam" not in file:
        print("file: " + file)
        # open csv file
        with open(file, 'r') as f:
            # read first line
            all_lines = f.readlines()
            # remove header
            all_lines.pop(0)
            tuples1 = [(float(thresh), float(se), float(se_boost)) for thresh, se, se_boost in [line.split(',') for line in all_lines]]
            control_se = tuples1[0][1]
            # sort by se_boost
            tuples1.sort(key=lambda tup: tup[2])
            # get highest se
            highest_se = tuples1[-1][1]
            # get highest se_boost
            highest_se_boost = tuples1[-1][2]
            # get threshold for highest se_boost
            highest_se_boost_threshold = tuples1[-1][0]
            # add train and test results to lists
            if file.startswith('train'):
                train_rows.append([file, control_se, highest_se, highest_se_boost, highest_se_boost_threshold])
            else:
                test_rows.append([file, control_se, highest_se, highest_se_boost, highest_se_boost_threshold])

# add train and test results to csv file
for i in range(len(train_rows)):
    parts = train_rows[i][0].split(".")[0].split('_')
    csv_writer.writerow([train_rows[i][0], 'train', parts[-2], parts[-3], parts[-1], train_rows[i][1], train_rows[i][2], train_rows[i][3], train_rows[i][4]])
    csv_writer.writerow([test_rows[i][0], 'test', parts[-2], parts[-3], parts[-1], test_rows[i][1], test_rows[i][2], test_rows[i][3], test_rows[i][4]])
    # write blank line
    csv_writer.writerow([])

results.close()

            