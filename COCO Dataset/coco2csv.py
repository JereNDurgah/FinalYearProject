import os
import csv
import numpy as np
import pycocotools.coco as coco

# Load the COCO dataset
dataDir = '/COCO Dataset\Sheep\Sheep'
dataType = 'val2017'
annFile = 'COCO Dataset\Sheep\Sheep.json'.format(dataDir, dataType)
coco = coco.COCO(annFile)

# Get all image IDs
imgIds = coco.getImgIds()

# Open a CSV file to write the data to
file_name = 'Sheep.csv'
with open(file_name, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Write the header row
    writer.writerow(['file_name', 'annotation_id', 'x1', 'y1', 'x2', 'y2'])
    
    # Loop over all image IDs
    for imgId in imgIds:
        # Get all annotations for the current image
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        # Get the file name for the current image
        img = coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        file_name = file_name[10:].strip('\\')
        #use [10:] for Goat and Sheep 
        #use [9:] for Dog, Cat, Cow
        #print(file_name)

        
        # Loop over all annotations for the current image
        for ann in anns:
            # Get the annotation ID and bounding box coordinates
            annotation_id = ann['id']
            x1, y1, width, height = ann['bbox']
            x2 = x1 + width
            y2 = y1 + height
            
            # Write the data to the CSV file
            writer.writerow([file_name, annotation_id, x1, y1, x2, y2])




