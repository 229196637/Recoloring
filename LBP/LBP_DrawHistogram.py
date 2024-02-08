import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

import os
import matplotlib.pyplot as plt
from libsvm.svmutil import *

def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def drawHistogram(data):
    x_num, y_num = data.shape
    k = np.arange(x_num)
    y = np.zeros(x_num)
    for j in range(x_num):
        y[j] = data[j][1]


    plt.plot(k, y, marker='o', linestyle='', color='b')
    plt.title('dx2(G)')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


path = "HistogramData/rec"

files = [f for f in listdir(path) if isfile(join(path, f))]

for n in range(len(files)):
    data = np.loadtxt(join(path, files[n]))
    data /= 255
    drawHistogram(data)

