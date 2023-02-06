import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import faster_rcnn
import os

# Load the 5 COCO datasets

#Path to all files
coco_root = "FinalYearProject\COCO Dataset"

#coco_datasets indiciates the final section of the path required
coco_datasets = ["Cat.json", "Dog.json", "Cow.json", "Goat.json", "Sheep.json"]

#coco_train_datasets uses a comprehension list to  produce a valid file path to each COCO dataset.json
coco_train_datasets = [os.path.join(coco_root, f) for f in coco_datasets]


print(coco_train_datasets)

# Load the custom COCO dataset
train_dataset = [..., ..., ...]
val_dataset = [..., ..., ...]

# Load the ResNet-50 backbone
backbone = keras.applications.resnet.ResNet50(weights='imagenet',include_top=False,input_tensor=Input(shape=(None, None, 3)))

# Initialize the Faster R-CNN model
model = faster_rcnn.build_faster_rcnn(num_classes=5, backbone='ResNet50')

# Compile the model
model.compile(optimizer=Adam(lr=0.005), loss=[faster_rcnn.rpn_loss, faster_rcnn.class_loss])

# Callbacks to ensure efficient training
checkpoint = ModelCheckpoint('faster_rcnn_resnet50.h5', verbose=1, save_best_only=True)
tensorboard = TensorBoard(log_dir='logs')
early_stopping = EarlyStopping(patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=1)
callbacks = [checkpoint, tensorboard, early_stopping, reduce_lr]

# Train the model on the 5 datasets
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=callbacks)

# Save the trained model
model.save('faster_rcnn_resnet50.h5')