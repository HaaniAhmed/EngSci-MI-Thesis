import os

path = "C:\\Users\\George\\OneDrive\\Documents\\University\\Y4F\\Thesis\\thesis-rl-project-main\\thesis-rl-project-main\\src\\results\\drl\\CartPole-v0"
runs = os.listdir(path)

# recursively delete folder
def delete_folder(path):
    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                delete_folder(item_path)
            else:
                os.remove(item_path)
        os.rmdir(path)
    


total_before = 0
for run in runs:
    # get number of subdirectories
    subdirs = os.listdir(path + "\\" + run)
    subdirs.sort()
    num_subdirs = len(subdirs)
    total_before += num_subdirs
    if len(subdirs) < 100:
        # delete entire run
        print("Deleting " + run)
        delete_folder(path + "\\" + run)


total_after = 0
runs = os.listdir(path)
for run in runs:
    # get number of subdirectories
    subdirs = os.listdir(path + "\\" + run)
    num_subdirs = len(subdirs)
    total_after += num_subdirs

print("Total before: " + str(total_before))
print("Total after: " + str(total_after))



