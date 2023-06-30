import numpy as np
import os
import shutil

input_folder = 'data'
output_folder = 'output'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

file_list = os.listdir(input_folder)

np.random.shuffle(file_list)

total_files = len(file_list)
train_size = int(train_ratio * total_files)
val_size = int(val_ratio * total_files)

train_files = file_list[:train_size]
val_files = file_list[train_size:train_size + val_size]
test_files = file_list[train_size + val_size:]

os.makedirs(output_folder, exist_ok=True)
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'val')
test_folder = os.path.join(output_folder, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for file in train_files:
    shutil.move(os.path.join(input_folder, file), os.path.join(train_folder, file))
for file in val_files:
    shutil.move(os.path.join(input_folder, file), os.path.join(val_folder, file))
for file in test_files:
    shutil.move(os.path.join(input_folder, file), os.path.join(test_folder, file))
