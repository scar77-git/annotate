import albumentations as A
import cv2
import os
img = cv2.imread(os.path.join('D:/draw-YOLO-box/raw_images','1.jpg'))
if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#box=[[0.2470371198568873,0.10196779964221823,0.2747093023255814,0.20035778175313057],[0.6812388193202146,0.16189624329159213,0.4246422182468693,0.29159212880143115]]
if os.stat('D:/CAT_PROJ/labels/1.txt').st_size>0:
      file1 = open('D:/CAT_PROJ/labels/1.txt', 'r')
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
print(box)
class_labels=['body','refill']
transform = A.Compose([
      A.HorizontalFlip(),
      ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
transformed = transform(image=img, bboxes=box,class_labels=class_labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
print(transformed_bboxes)