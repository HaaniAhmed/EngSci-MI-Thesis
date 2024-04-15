import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


# loop through activations folder
act_folder = "activations_cartpole/"
act_files = os.listdir(act_folder)
act_files = [act_folder + f for f in act_files if f.endswith(".csv")]

hist_dir = "hist_cartpole/"

chunks = 2000

iterations = np.arange(0, 10000000, chunks)
lists = [[] for i in iterations]

# create dataframe
df = pd.DataFrame(columns=["num_iterations", "layer1", "layer2"])

def add_to_lists(x):
    # print(len(lists))
    # print(x.index[0])
    # print(x[1])
    lists[x.index[0]//chunks] += x[1].values.tolist()
    

# loop through files
for i,f in enumerate(act_files):
    print(f)
    # read in file
    df_temp = pd.read_csv(f, header=None).fillna(0)
    print(df_temp)
    df_temp.groupby(df_temp.index // chunks).apply(add_to_lists)

for i, vals in enumerate(lists):
    print(i, len(vals))

    if len(vals) == 0:
        print('empty')
        continue
    plt.hist(vals, bins=15, range=[0.0, 4.0])
    plt.title("Histogram of Entropy Values for Iteration " + str(i*chunks))
    plt.xlabel("Entropy Value")
    plt.ylabel("Frequency")

    plt.savefig(hist_dir + "hist_" + str(i*chunks) + ".png")
    print("saved hist_" + str(i*chunks) + ".png")
    plt.clf()

# print(df.head(), df.shape)
# df.plot(y="num_iterations", x="layer1", kind="scatter")
# df.plot(y="num_iterations", x="layer2", kind="scatter")

# plt.show()