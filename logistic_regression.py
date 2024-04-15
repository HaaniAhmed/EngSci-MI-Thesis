# import csv and use logistic regression to predict success

import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from sklearn import linear_model
from sklearn.linear_model import *
from sklearn.neural_network import *
from sklearn.metrics import accuracy_score
import pandas as pd

# collect more metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

TRAIN_FILE_NAME = "relu_train_gradcam.csv"
# TEST_FILE_NAME = "relu_test_gradcam.csv"

# TRAIN_FILE_NAME = "tanh_train_gradcam.csv"
# TEST_FILE_NAME = "tanh_test_gradcam.csv"

# TRAIN_FILE_NAME = "tanh_v1_train_gradcam.csv"
# TEST_FILE_NAME = "tanh_v1_test_gradcam.csv"

# TRAIN_FILE_NAME = "tanh_v0_all_gradcam.csv"

# TRAIN_FILE_NAME = "relu_v0_train_gbp.csv"

# read in data
with open(TRAIN_FILE_NAME, 'r') as f:
    reader = csv.reader(f)

    # pandas
    df = pd.read_csv(f)
    # plot every column agaisnt 2nd last
    # ax = df.plot(x=df.columns[-2], y=df.columns[0], style='o')
    # # df.plot(x=df.columns[-2], y=df.columns[1], style='o', ax=ax)
    # # df.plot(x=df.columns[-2], y=df.columns[2], style='o', ax=ax)
    # # df.plot(x=df.columns[-2], y=df.columns[3], style='o', ax=ax)
    # # df.plot(x=df.columns[-2], y=df.columns[4], style='o', ax=ax)
    # # df.plot(x=df.columns[-2], y=df.columns[5], style='o', ax=ax)
    # ax.set_xlim(0, 20000)
    # plt.show()

    overall_entropy = df['overall_entropy'].to_numpy()
    grad_entropy = df['grad_entropy'].to_numpy()
    l1_entropy = df['l1_entropy'].to_numpy()
    l1_entropy_grad = df['l1_grad_entropy'].to_numpy()
    l2_entropy = df['l2_entropy'].to_numpy()
    l2_entropy_grad = df['l2_grad_entropy'].to_numpy()
    # successes = df['success'].to_numpy()

    # # print success rate
    # print('success rate: ', np.mean(successes))

    rew_steps = df['num_steps'].to_numpy()
    rew_steps = np.where(rew_steps == -1, 100000, rew_steps)

    # treat success as <= 3000 steps
    success_thresh = 5000
    success = np.where(rew_steps <= success_thresh, 1, 0)
    print('success rate: ', np.mean(success))

    # create (entropy, reward steps) tuples
    # overall_entropy_pairs = [(x, y) for x, y in zip(overall_entropy, rew_steps)]


    # select all data where entropy >= 1.3
    entropy = 1.05
    rew_steps_thresh = rew_steps[np.where(overall_entropy >= entropy)]
    print('success rate for entropy >= {}: {}'.format(entropy, np.mean(rew_steps_thresh <= success_thresh)))
    


    # bin the entropy values and get average reward steps for each bin
    # average_reward_steps = []
    # entropy_bins = np.arange(0, 2, 0.1)
    # prev = 0
    # for i in entropy_bins:
    #     # get mean of reward steps only for entropy
    #     steps = rew_steps[np.where((overall_entropy >= prev) & (overall_entropy < i))]
    #     avg_steps = np.mean(steps)
    #     print(i, avg_steps, len(steps))
    #     average_reward_steps.append(avg_steps)
    #     prev = i
    
    # # plot average reward steps vs entropy
    # plt.plot(entropy_bins, average_reward_steps)
    # plt.xlabel('entropy')
    # plt.ylabel('average reward steps')
    # plt.show()

    # plot scatter of entropy vs reward steps
    plt.scatter(overall_entropy, rew_steps, cmap='bwr', c=success, alpha=0.5)
    # horizontal line at 3000
    # plt.axhline(y=3000, color='r', linestyle='-')
    plt.axvline(x=entropy, color='r', linestyle='-')
    plt.xlabel('Entropy')
    plt.ylabel('Reward Steps')
    plt.show()

    # bin rew_steps and plot averahe entropy for each bin
    # average_entropy = []
    # steps = np.arange(0, 20000, 1000)
    # prev = 0
    # for i in steps:
    #     # get mean of entropy only for step
    #     avg_ent = np.mean(grad_entropy[np.where((rew_steps >= prev) & (rew_steps < i))])
    #     print(i, avg_ent)
    #     average_entropy.append(avg_ent)
    #     prev = i

    # plt.plot(steps, average_entropy)
    # plt.show()

    
    # train_data = list(reader)
    # # remove header
    # train_data = train_data[1:]

    # # train on layer 1 and layer 2 as separate features with grads separated
    # train_x = [row[5:6] for row in train_data]
    # # turn all data into floats
    # train_x = [[float(x) for x in row] for row in train_x]

    # # train_y = [int(row[-1] == 'True') for row in train_data]
    # rew_steps = [int(row[-2]) if int(row[-2]) != -1 else 100000 for row in train_data]
    # limit = np.median(rew_steps)
    # train_y = [int(row) < limit for row in rew_steps]
    # train_y = rew_steps

    # plot data
    # plt.scatter(train_x, rew_steps, c=train_y, cmap='bwr')
    # plt.show()

    # train regression neural network
    # model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='adam', max_iter=1000)
    train_x = grad_entropy
    train_x = train_x.reshape(-1, 1)
    train_y = success
    model = LogisticRegression()
    model.fit(train_x, train_y)

    # plot regression line
    plt.scatter(train_x, train_y, c=train_y, cmap='bwr')
    plt.plot(train_x, model.predict(train_x), color='black')
    plt.show()

    # test on test data
    # with open(TEST_FILE_NAME, 'r') as f:
    #     reader = csv.reader(f)
    #     test_data = list(reader)
    #     # remove header
    #     test_data = test_data[1:]

    #     test_x = [row[5:6] for row in test_data]
    #     # turn all data into floats
    #     test_x = [[float(x) for x in row] for row in test_x]

    #     # test_y = [int(row[-1] == 'True') for row in test_data]
    #     # test_y = [int(row[-2]) for row in test_data]
    #     rew_steps = [int(row[-2]) if int(row[-2]) != -1 else 100000 for row in test_data]
    #     # limit = 10000
    #     test_y = [int(row) < limit for row in rew_steps]
    #     # test_y = rew_steps
    #     print(test_y)

    #     # predict
    #     pred_y = model.predict(test_x)
    #     # print(pred_y)

    #     # plot data
    #     # plt.scatter(test_x, test_y, c='b', label='actual')
    #     # plt.scatter(test_x, pred_y, c='r', label='predicted')
    #     # plt.legend()
    #     # plt.show()


    #     # calculate accuracy
    #     print(accuracy_score(test_y, pred_y))
    #     print(confusion_matrix(test_y, pred_y))
        # print(np.mean(np.abs(pred_y - test_y)))


