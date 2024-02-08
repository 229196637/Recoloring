import cv2
import numpy as np
import skimage.feature
import skimage.segmentation

from skimage import feature


def ULBP(source):
    height, width = source.shape[:2]
    dst = np.zeros((height, width, 3), dtype=np.uint8)

    lbp_value = np.zeros((8, 3), dtype=np.uint8)
    neighbours = np.zeros((8, 3), dtype=np.uint8)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            center = source[row, col]

            neighbours[0] = source[row - 1, col - 1]
            neighbours[1] = source[row - 1, col]
            neighbours[2] = source[row - 1, col + 1]
            neighbours[3] = source[row, col + 1]
            neighbours[4] = source[row + 1, col + 1]
            neighbours[5] = source[row + 1, col]
            neighbours[6] = source[row + 1, col - 1]
            neighbours[7] = source[row, col - 1]

            # 领域数值判断
            for i in range(8):
                for j in range(3):
                    if neighbours[i, j] > center[j]:
                        lbp_value[i, j] = 1
                    else:
                        lbp_value[i, j] = 0

            # 转成二进制数
            lbp = np.zeros(3, dtype=np.uint8)
            for k in range(8):
                lbp += lbp_value[k] * (2 ^ k)
            for k in range(3):
                if(getHopCnt(lbp[k]) >2):
                    lbp[k] = 58

            dst[row, col] = lbp

    return dst


def CircularLBP(source, P=8, R=1.0, method='default'):
    gray = cv2.cvtColor(source, code=cv2.COLOR_BGR2GRAY)
    CircularLBP = skimage.feature.local_binary_pattern(gray, P, R, method)
    CircularLBP = CircularLBP.astype(np.uint8)
    return CircularLBP


def ulbp(source, P=8, R=1.0):
    gray = cv2.cvtColor(source, code=cv2.COLOR_BGR2GRAY)
    CircularLBP = skimage.feature.local_binary_pattern(gray, P, R, 'default')
    CircularLBP = CircularLBP.astype(np.uint8)

    height, width = source.shape[:2]

    for x in range(0, width - 1):
        for y in range(0, height - 1):
            if getHopCnt(CircularLBP[x, y]) > 2:
                CircularLBP[x, y] = P * (P - 1) + 2

    CircularLBP = cv2.cvtColor(CircularLBP, code=cv2.COLOR_GRAY2BGR)
    return CircularLBP


def getHopCnt(num, P=8):
    '''
    :param num:8位的整形数，0-255
    :return:
    '''
    if num > (2 ** P) - 1:
        num = (2 ** P) - 1
    elif num < 0:
        num = 0

    num_b = bin(num)
    num_b = str(num_b)[2:]

    # 补0
    if len(num_b) < P:
        temp = []
        for i in range(P - len(num_b)):
            temp.append('0')
        temp.extend(num_b)
        num_b = temp

    cnt = 0
    for i in range(P):
        if i == 0:
            former = num_b[-1]
        else:
            former = num_b[i - 1]
        if former == num_b[i]:
            pass
        else:
            cnt += 1

    return cnt
