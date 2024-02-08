import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import os

save_path = "Txt/data.txt"
def make_histogram(source, rec, n,P=8):
    height, width, shape = source.shape[:3]
    K = P * (P - 1) + 2
    H_source = np.zeros((K, shape), dtype=np.uint8)
    H_rec = np.zeros((K, shape), dtype=np.uint8)
    for k in range(K):
        print(k)
        for x in range(0, height - 1):
            for y in range(0, width - 1):
                for s in range(3):
                    H_source[k, s] += calculate_f(source[x][y][s], k)
                    H_rec[k, s] += calculate_f(rec[x][y][s], k)
    nat_path = 'HistogramData/net/data-{}.txt'.format(n)
    rec_path = 'HistogramData/rec/data-{}.txt'.format(n)
    np.savetxt(rec_path,H_rec,fmt='%.3f')
    np.savetxt(nat_path,H_source,fmt='%.3f')
    dist = DIST(H_rec, H_source)
    dx2 = DX2(H_rec, H_source)
    save_path = 'Txt/data-{}.txt'.format(n)
    np.savetxt(save_path, dist, fmt='%.3f')
    save_path2 = 'Txt/dx2-{}.txt'.format(n)
    np.savetxt(save_path2, dx2, fmt='%.3f')
    return dist, dx2


def calculate_f(x, y):
    if x == y:
        return 1
    else:
        return 0


def DIST(H_rec, H_nat):
    distance = np.sqrt(np.square((H_rec - H_nat)))
    return distance


def DX2(H_rec, H_nat, P=8):
    epsilon = 1e-10
    K = P * (P - 1) + 2
    dx2 = np.zeros(3, dtype=np.float64)
    for k in range(K):
        for s in range(3):
            denominator = (H_rec[k][s] + H_nat[k][s])
            if abs(denominator) > epsilon:
                dx2[s] += (1 / 2) * ((H_rec[k][s] - H_nat[k][s]) ** 2) / denominator
    return dx2


image_path = "LBP_Image/"
reco_path = "LBP_Reco/"

image = [f for f in listdir(image_path) if isfile(join(image_path, f))]
reco = [f for f in listdir(reco_path) if isfile(join(reco_path, f))]
test = np.zeros((6, 3))

for n in range(1):
    im = cv2.imread(join(image_path, image[n]))
    rc = cv2.imread(join(reco_path, reco[n]))

    Dist, dx2 = make_histogram(im, rc, n)
#    nat_path = 'HistogramData/net/data-{}.txt'.format(n)
#    rec_path = 'HistogramData/rec/data-{}.txt'.format(n)
# np.savetxt(rec_path,H_rec,fmt='%.3f')
# np.savetxt(nat_path,H_nat,fmt='%.3f')
# save_path = 'Txt/data-{}.txt'.format(n)
# dx2 = dx2(H_rec, H_nat)
# dist = DIST(H_rec, H_nat)
# np.savetxt(save_path, dist, fmt='%.3f')
