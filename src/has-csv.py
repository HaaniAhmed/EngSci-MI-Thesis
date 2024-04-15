import os

path = "C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\drl\\CartPole-v0"
runs = os.listdir(path)

present = 0

for run in runs:
    # check if run has csv
    if not os.path.isfile(path + "\\" + run + "\\" + run + ".xlsx"):
        print("Missing xlsx for " + run)
    else:
        present += 1

print("Total runs: " + str(len(runs)))
print("Total with xlsx: " + str(present))