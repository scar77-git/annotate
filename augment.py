from imgaug import augmenters as iaa
from PIL import Image
import glob
import imageio
import imgaug as ia
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import boto3
import cv2
import os
from os import path
import pandas as pd
import shutil

# Fetching image from s3
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

def savelabel(bbox1,pos):
    pos1=str(pos)+'.txt'
    address='D:/draw-YOLO-box/Augmented_labels/{}'.format(pos1)
    j=0
    data=[]
    for x in bbox1:
        str1=str(j)+" "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])
        data.append(str1)
        j=j+1
    file1 = open(address, 'w')
    for d in data:
        file1.write(d+"\n")
    file1.close()  
def saveImage(image,address,pos):
    #imgSaveDir = path.join(address,pos+'.jpg')
     imgSaveDir = address+'/'+pos+'.jpg'
     cv2.imwrite(imgSaveDir , image)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images
image_list = load_images_from_folder('D:/draw-YOLO-box/raw_images')
i=1
for image1 in image_list:
  i=i+1
  print(i)
  tname=str(i)+'.txt'
  iname=str(i)+'.jpg'
  label_path='D:/draw-YOLO-box/labels/{}'.format(tname)
  img_path='D:/draw-YOLO-box/raw_images/{}'.format(iname)
  height, width, channels = image1.shape
  if os.stat(label_path).st_size>0:
      shutil.copy(img_path,'D:\draw-YOLO-box\Augmented')
      shutil.copy(label_path,'D:\draw-YOLO-box\Augmented_labels')
      file1 = open(label_path, 'r')
      Lines = file1.readlines()
      count = 0
      box=[]
      for line in Lines:
        count += 1
        l=line.split(' ')
        del l[0]
        del l[4]
        l=list(map(float,l)) 
        box.append(l)
    
      # Using albumentations
      print(box)
      if len(box)>1:
          class_labels = ['body', 'refill']
      elif len(box)==1:
          class_labels=['refill']
          
     # Horizontal flip
     
      transform = A.Compose([
      A.HorizontalFlip(p=0.5),
      ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
      transformed = transform(image=image1, bboxes=box,class_labels=class_labels)
      transformed_image = transformed['image']
      transformed_bboxes = transformed['bboxes']
      pos=str(i)+'_'+str(1)
      pos1=str(i)+'_'+str(1)
      saveImage(transformed_image,'D:/draw-YOLO-box/Augmented',pos)
      savelabel(transformed_bboxes,pos1)
      #transformed_class_labels = transformed['class_labels']
      
      # rotate
      
      transform = A.Compose([
      A.RandomRotate90(),
      ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
      transformed = transform(image=image1, bboxes=box, class_labels=class_labels)
      transformed_image = transformed['image']
      transformed_bboxes = transformed['bboxes']
      pos=str(i)+'_'+str(2)
      pos1=str(i)+'_'+str(2)
      saveImage(transformed_image,'D:/draw-YOLO-box/Augmented',pos)
      savelabel(transformed_bboxes,pos1)
      #transformed_class_labels = transformed['class_labels']
      
      # Random brightness and contrast
      
      transform = A.Compose([
      A.RandomBrightnessContrast(p=0.2),
      ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
      transformed = transform(image=image1, bboxes=box,class_labels=class_labels)
      transformed_image = transformed['image']
      transformed_bboxes = transformed['bboxes']
      pos=str(i)+'_'+str(3)
      pos1=str(i)+'_'+str(3)
      saveImage(transformed_image,'D:/draw-YOLO-box/Augmented',pos)
      savelabel(transformed_bboxes,pos1)
      #transformed_class_labels = transformed['class_labels']
      
      # Heu Saturation
      
      transform = A.Compose([
      A.HueSaturationValue(p=1),
      ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
      transformed = transform(image=image1, bboxes=box,class_labels=class_labels )
      transformed_image = transformed['image']
      transformed_bboxes = transformed['bboxes']
      pos=str(i)+'_'+str(4)
      pos1=str(i)+'_'+str(4)
      saveImage(transformed_image,'D:/draw-YOLO-box/Augmented',pos)
      savelabel(transformed_bboxes,pos1)
      #transformed_class_labels = transformed['class_labels']
      
      
      """
      #Using IMGAUG:
      #============
      c=iaa.Crop(px=(1, 16), keep_size=False)
      images_aug,bbs_aug=c(image=image1,bounding_boxes=box)
      pos=str(i)+'_'+str(1)
      print(pos)
      #saveImage(images_aug,'D:/draw-YOLO-box/Augmented',pos)
      f=iaa.Fliplr(0.5)
      images_aug,bbs_aug=f(image=image1,bounding_boxes=box)
      pos=str(i)+'_'+str(2)
      print(pos)
      #saveImage(images_aug,'D:/draw-YOLO-box/Augmented',pos)
      g=iaa.GaussianBlur(sigma=(0, 3.0))
      images_aug,bbs_aug=g(image=image1,bounding_boxes=box)
      pos=str(i)+'_'+str(3)
      print(pos)
      #saveImage(images_aug,'D:/draw-YOLO-box/Augmented',pos)
      ro=iaa.Affine(rotate=5)
      images_aug,bbs_aug=ro(image=image1,bounding_boxes=box)
      pos=str(i)+'_'+str(4)
      print(pos)
      #saveImage(images_aug,'D:/draw-YOLO-box/Augmented',pos)
      """
      
      """
       l=line.split()
       class_idx = int(staff[0])
       x_center, y_center, w, h = float(staff[1])*width, float(staff[2])*height, float(staff[3])*width, float(staff[4])*height
       x1 = round(x_center-w/2)
       y1 = round(y_center-h/2)
       x2 = round(x_center+w/2)
       y2 = round(y_center+h/2)   
       l=[x1,y1,x2,y2]
       box.append(l)
       with open(label_path, 'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if line.strip()]  # Remove empty lines
        annotations = np.array([line.split() for line in lines], dtype=np.float32)
       #annotations = np.loadtxt(label_path, dtype=np.float32,delimiter='\n')
       del annotations[0][0]
       del annotations[1][0]
    """
      
      
  