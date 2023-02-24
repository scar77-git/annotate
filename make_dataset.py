import os
import shutil
directory = "dataset"
path = "D:/draw-YOLO-box/dataset"
#path = os.path.join(parent_dir, directory)
os.mkdir(path)
path1="D:/draw-YOLO-box/dataset/training"
#parent_dir=path
#path_org = os.path.join(parent_dir, directory)
os.mkdir(path1)
source_dir="D:/draw-YOLO-box/Augmented"
destination_dir="D:/draw-YOLO-box/dataset/training/images"
shutil.copytree(source_dir, destination_dir)

source_dir="D:/draw-YOLO-box/Augmented_labels"
destination_dir="D:/draw-YOLO-box/dataset/training/labels"
shutil.copytree(source_dir, destination_dir)