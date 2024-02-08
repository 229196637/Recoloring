import sys

import numpy as np
import cv2


def color_transfer(source, target):
    # convert color space from BGR to L*a*b color space
    ## L* for the lightness from black to white, a* from green to red, and b* from blue to yellow.
    # note - OpenCV expects a 32bit float rather than 64bit
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color stats for both images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # split the color space
    (l, a, b) = cv2.split(target)

    # substarct the means from target image
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviation
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clipping the pixels between 0 and 255(0 denotes black and 255 denotes white)
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels
    transfer = cv2.merge([l, a, b])

    # converting back to BGR
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    return transfer


# In[3]:


def image_stats(image):
    # compute mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    return (lMean, lStd, aMean, aStd, bMean, bStd)


# In[4]:

# show the image in windows.
def show_image(title, image, width=720):
    r = width / float(image.shape[1])
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow(title, resized)

def progress_bar(finish_tasks_number, tasks_number):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


from os import listdir
from os.path import isfile, join

path = "E:\\Project\\pythonProject1\\ImaNet\\val"
path_original = "E:\\Project\\pythonProject1\\ImaNet\\original"
path_recoloring = "E:\\Project\\pythonProject1\\ImaNet\\recoloring"
path_train = "E:\\Project\\pythonProject1\\ImaNet\\trainingset"
onlyfiles1 = [f for f in listdir(path) if isfile(join(path, f))]
img1 = np.empty(len(onlyfiles1)//2,dtype=object)
img2 = np.empty(len(onlyfiles1)//2,dtype=object)
print(len(onlyfiles1)) #导入图片数量

# 前一半作为source
# 后一半作为target
for n in range(len(onlyfiles1)):
    img1[n] = cv2.imread(join(path,onlyfiles1[n]))
    img1[n] = cv2.cvtColor(img1[n], cv2.COLOR_BGR2RGB)
    img1[n]=cv2.resize(img1[n],(500,500))
    img2[n] = cv2.imread(join(path,onlyfiles1[len(onlyfiles1) - n-1]))
    img2[n] = cv2.cvtColor(img2[n], cv2.COLOR_BGR2RGB)
    img2[n]=cv2.resize(img2[n],(500,500))

    transfer = color_transfer(img1[n],img2[n])

    cv2.imwrite(join(path_original,'img.{}.jpg'.format(n)),img2[n])
    cv2.imwrite(join(path_recoloring, 'img.{}.jpg'.format(n)), transfer)

    progress_bar(n,len(onlyfiles1))


onlyfiles1 = [ f for f in listdir(path_original) if isfile(join(path_original,f)) ]
onlyfiles2 = [ f for f in listdir(path_recoloring) if isfile(join(path_recoloring,f)) ]
img1 = np.empty(len(onlyfiles1),dtype=object)
img2 = np.empty(len(onlyfiles2),dtype=object)

for n in range(len(onlyfiles1)):
    img1[n] = cv2.imread(join(path_original,onlyfiles1[n]))
    img1[n] = cv2.cvtColor(img1[n], cv2.COLOR_BGR2RGB)
    img1[n]=cv2.resize(img1[n],(500,500))

    img2[n] = cv2.imread(join(path_recoloring,onlyfiles2[n]))
    img2[n] = cv2.cvtColor(img2[n], cv2.COLOR_BGR2RGB)
    img2[n]=cv2.resize(img2[n],(500,500))

    cv2.imwrite(join(path_train, 'img.{}.jpg'.format(n)), img1[n])
    cv2.imwrite(join(path_train, 'pic.{}.jpg'.format(n)), img2[n])

    progress_bar(n, len(onlyfiles1))