import cv2
import numpy as np

import LBP


# show the image in windows.
def show_image(title, image, width=720):
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow(title, resized)


def make_histogram(image, P=8):
    height, width, shape = source.shape[:3]
    K = P * (P - 1) + 2
    H = np.zeros((K, shape), dtype=np.uint8)
    for k in range(K):
        print(k)
        for x in range(0, height - 1):
            for y in range(0, width - 1):
                for s in range(3):
                    H[k, s] += calculate_f(image[x][y][s], k)
    return H

def calculate_f(x, y):
    if x == y:
        return 1
    else:
        return 0


source = cv2.imread("E:\\Project\\pythonProject1\\ImaNet\\trainingset\\img.0.jpg")
recolor = cv2.imread("E:\\Project\\pythonProject1\\ImaNet\\recoloring\\img.0.jpg")
transform = LBP.ulbp(source)
transform2 = LBP.ulbp(recolor)

print(make_histogram(transform, 8))

show_image("scr", source)
show_image("source", transform)
show_image("Recolor", transform2)
cv2.waitKey(0)
