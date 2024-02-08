import os
import random
import shutil
from os import listdir
from os.path import isfile, join
import numpy as np

path = 'HistogramData/data'
path_label = "HistogramData/data_label"
train_path = 'HistogramData/data_test'
test_path = 'HistogramData/data_train'


def min_max_normalize(data):
    min_vals = np.min(data)
    max_vals = np.max(data)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def manage_data_1(manager,Filepath,targetFileName):
    test = 0
    libsvm_train_data = []
    for file_name in manager:
        with open(join(path_label,file_name), 'r') as file:
            raw_data = file.readlines()
            if file_name.split('_')[0] == 'org':
                all_in_line = '0' + ' '
            else:
                all_in_line = '1' + ' '
            i = 0
            for line in range(len(raw_data)):
                values = raw_data[line].strip().split()
                features = []
                for value in values:
                    features.append(f"{i + 1}:{value}")
                    i += 1
                libsvm_line = f" {' '.join(features)}"
                all_in_line += libsvm_line
        libsvm_train_data.append(all_in_line)


    # 将转换后的数据写入 LIBSVM 格式文件
    with open(join(Filepath,targetFileName), 'w') as file:
        file.write('\n'.join(libsvm_train_data))


def manage():
    files = [f for f in listdir(path) if isfile(join(path, f))]

    files_length = len(files) // 2
    rec = files[:files_length]
    org = files[files_length:]

    train_manage = random.sample(rec, 50) + random.sample(org, 50)
    test_manage = random.sample(rec, 30) + random.sample(org, 30)

    # 下面不要了

    files_length //= 2  # 修正了这一行
    train_manage = rec[:files_length] + org[:files_length]
    test_manage = rec[files_length:] + org[files_length:]

    train_manage = random.sample(train_manage, 100)
    test_manage = random.sample(test_manage, 60)

    i = 0
    libsvm_train_data = []
    # 将 train_manage 列表中的文件保存到 train_path 目录
    for file_name in train_manage:
        source_path = join(path, file_name)
        destination_path = join(train_path, file_name)
        shutil.copyfile(source_path, destination_path)

        # 归一化
        data = min_max_normalize(np.loadtxt(destination_path))
        np.savetxt(destination_path, data, fmt='%.3f')

        # 读取原始文本数据
        with open(destination_path, 'r') as file:
            raw_data = file.readlines()
            for line in range(len(raw_data)):
                values = raw_data[line].strip().split()
                features = [f"{i + 1}:{value}" for i, value in enumerate(values, start=0)]
                libsvm_line = f"{line + 1} {' '.join(features)}"
                libsvm_train_data.append(libsvm_line)
            # if line < len(libsvm_train_data):
            #    libsvm_line = f" {' '.join(features)}"
            # libsvm_train_data[line] = libsvm_train_data[line] + libsvm_line
            # else:
            #   libsvm_line = f"{line + 1} {' '.join(features)}"
            #  libsvm_train_data.append(libsvm_line)
    # 将转换后的数据写入 LIBSVM 格式文件
    with open('HistogramData/train_data/train.txt', 'w') as file:
        file.write('\n'.join(libsvm_train_data))

    # 将文件进行组合

    i = 0
    libsvm_train_data = []
    # 将 test_manage 列表中的文件保存到 test_path 目录
    for file_name in test_manage:
        source_path = join(path, file_name)
        destination_path = join(test_path, file_name)
        shutil.copyfile(source_path, destination_path)

        # 归一化
        data = min_max_normalize(np.loadtxt(destination_path))
        np.savetxt(destination_path, data, fmt='%.3f')

        # 读取原始文本数据
        with open(destination_path, 'r') as file:
            raw_data = file.readlines()
            for line in range(len(raw_data)):
                values = raw_data[line].strip().split()
                features = [f"{i + 1}:{value}" for i, value in enumerate(values, start=0)]
                libsvm_line = f"{line + 1} {' '.join(features)}"
                libsvm_train_data.append(libsvm_line)
    # 将转换后的数据写入 LIBSVM 格式文件
    with open('HistogramData/test_data/test.txt', 'w') as file:
        file.write('\n'.join(libsvm_train_data))

    # 可选地，如果需要，可以使用 np.savetxt() 保存数据
    # 例如：
    # np.savetxt(join(train_path, 'example.txt'), your_data_array)


# 处理一下标签
def manage_label(files):
    for file in files:
        number = int(file.split('-')[1].split('.')[0])
        if number < 5000:
            os.rename(join(path,file), join(path_label,'org_'+file))
        else:
            os.rename(join(path, file), join(path_label, 'rec_' + file))


# 调用 manage 函数执行文件管理
files = [f for f in listdir(path_label) if isfile(join(path_label, f))]


files_length = len(files) // 2
rec = files[:files_length]
org = files[files_length:]
rec_length = len(rec)
org_length = len(org)

train_manage = rec[:(rec_length-100)] + org[:(org_length-100)]
test_manage = rec[(rec_length-100):rec_length] + org[(org_length-100):org_length]



manage_data_1(train_manage,'HistogramData/train_data','train.txt')
manage_data_1(test_manage,'HistogramData/test_data','test.txt')