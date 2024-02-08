from os import listdir
from os.path import isfile, join
import os

import cv2

import LBP

Rec_path = "LBP_Reco/"
Img_path = "LBP_Image/"
targetFile = "../ImaNet/recoloring"
imgFile = "../ImaNet/trainingset"

rec_files = [f for f in listdir(targetFile) if isfile(join(targetFile, f))]
img_files = [f for f in listdir(imgFile) if isfile(join(imgFile, f))]

for n in range(1):
    source_rec = cv2.imread(join(targetFile,rec_files[n]))
    source_img = cv2.imread(join(imgFile,img_files[n]))

    source_rec = cv2.cvtColor(source_rec,code =cv2.COLOR_BGR2YCrCb)
    source_img = cv2.cvtColor(source_img,code =cv2.COLOR_BGR2YCrCb)

    transform_img = LBP.ULBP(source_img)
    transform_Rec = LBP.ULBP(source_rec)
    cv2.imwrite(os.path.join(Rec_path , 'rec.{}.jpg'.format(n)),transform_Rec)
    cv2.imwrite(os.path.join(Img_path , 'Img.{}.jpg'.format(n)),transform_img)
