import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np
import cv2
import os

batch_size = 32
epochs = 100
learning_rate = 1e-4



dataset_folder = os.path.join(os.getcwd(),"COCO-Datasets")

object_folder = ["Cat", "Cow", "Dog", "Goat", "Sheep"]

for o in range(len(object_folder)):
    annotation_path = os.path.join(dataset_folder,object_folder[o],object_folder[o]+".csv")
    csv_file = open(annotation_path).read().strip().split("\n")
    
    data = []
    targets = []
    filenames = []
    
    for idx,row in enumerate(csv_file):
        if idx == 0:
            continue
        else:
            row = row.split(",")
            (fileName, annotation_id , xStart, yStart, xEnd, yEnd) = row

        file_path = os.path.join(dataset_folder,object_folder[o],object_folder[o],fileName)
        
        image_data = cv2.imread(file_path)
       
    
        height,width,depth = image_data.shape
        
        startX = float(xStart) / width
        startY = float(yStart) / height
        
        endX = float(xEnd) / width
        endY = float(yEnd) / height
        
        image = load_img(file_path)
        image = img_to_array(image)
        
        data.append(image)
        targets.append((startX, startY, endX, endY))
        filenames.append(file_path)
        
        data = np.array(data,dtype=np.float32)/255.0
        targets = np.array(targets, dtype=np.float32)
        
        split = train_test_split(data, targets, filenames, test_size=0.10, train_size = 0.90,random_state=42)