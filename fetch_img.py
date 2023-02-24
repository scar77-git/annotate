import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import boto3
import cv2
import os
from PIL import Image
s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket('layerxannotate')
for i in range(20):
 file_name=str(i+1)+'.jpg'
 tfile_name='1.0.1_'+str(i+1)+'.txt'
 path='dataset/63e33130ea891adb27686486/63e3298ff0d9a54fe149c829/{}/{}'.format(i+1,file_name)
 object = bucket.Object(path)
 os.chdir(r"D:\draw-YOLO-box\raw_images")
 object.download_file(file_name)
 path='dataset/63e33130ea891adb27686486/63e3298ff0d9a54fe149c829/{}/{}'.format(i+1,tfile_name)
 object = bucket.Object(path)
 os.chdir(r"D:\draw-YOLO-box\labels")
 object.download_file(tfile_name)
 nfile=str(i+1)+'.txt'
 os.rename(tfile_name,nfile)
 os.chdir(r"D:\draw-YOLO-box\labels_csv")
 object.download_file(tfile_name)
 nfile=str(i+1)+'.csv'
 os.rename(tfile_name,nfile)
#image = cv2.imread('1_thumbnail.jpg')
#cv2.imshow('image window', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
