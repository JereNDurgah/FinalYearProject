from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input,Flatten, Dense
from keras.optimizers import Adam

from keras.applications import ResNet50V2
from keras.models import Model

from keras.utils import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
gpu_device = "/device:GPU:0"
cpu_device = "/device:CPU:0"

dataClasses = {
    0 : "Cat",
    1 : "Cow",
    2 : "Dog",
    3 : "Goat",
    4 : "Sheep"
}
dataset_directory = os.path.join(os.getcwd(),"COCO Dataset")
logs_directory = os.path.join(os.getcwd(),"Logs")
models_directory = os.path.join(os.getcwd(),"Model")

trainParams = {
    "Batch-Size": 4,
    "Epochs": 100,
    "Learning-Rate":1e-4,
    "Target-Size": (720,720)
}

def Model_AI(input_shape, num_classes):
    # load the ResNet50V2 network, ensuring the head FC layers are left off
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=input_shape)

    # freeze all VGG layers so they will *not* be updated during the training process
    base_model.trainable = False

    # flatten the max-pooling output of VGG
    flatten = base_model.output
    flatten = Flatten()(flatten)

    # construct a fully-connected layer header to output the predicted bounding box coordinates
    x = Dense(128, activation="relu")(flatten)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(5, activation="softmax")(x)

    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=base_model.input, outputs=x)

    # initialize the optimizer, compile the model, and show the model
    # Adam(learning_rate=trainParams['Learning-Rate'])
    model.compile(optimizer= Adam(learning_rate=trainParams['Learning-Rate']), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def Setup_Callbacks(modelname : str = "ResNet50V2_best.h5"):
    # TensorBoard for monitoring the training process
    tensorboard = TensorBoard(log_dir=logs_directory)

    # ModelCheckpoint to save the best model
    checkpoint = ModelCheckpoint(filepath=os.path.join(models_directory,modelname), save_best_only=True, verbose=1,monitor="val_accuracy")

    # EarlyStopping to stop training if there's no improvement in the validation loss
    early_stopping = EarlyStopping(monitor = "val_loss", patience=10, verbose=1)

    callbacks = [tensorboard, checkpoint, early_stopping]
    return callbacks
    

if __name__ == "__main__":
    # img_path = os.path.join(os.getcwd(),"1d12b087-373.png")
    # im = cv2.imread(img_path)
    # print(im.shape)
    
    #  Setting up the callback functions
    callbacks = Setup_Callbacks()

    # Preparing the data for training
    dataGenerator = ImageDataGenerator(validation_split=0.2,rescale=1./255)
    
    trainGenerator = dataGenerator.flow_from_directory(
        directory=dataset_directory,
        subset="training",
        target_size=trainParams['Target-Size'],
        batch_size=trainParams['Batch-Size']
    )
    valGenerator = dataGenerator.flow_from_directory(
        directory=dataset_directory,
        subset="validation",
        target_size=trainParams['Target-Size'],
        batch_size=trainParams['Batch-Size']
    )
    
    numClasses = len(trainGenerator.class_indices)
    inputShape = trainParams['Target-Size'] + (3,)
    
    # Getting the model
    model = Model_AI(input_shape=inputShape,num_classes=numClasses)
    
    with tf.device(cpu_device):
        model.fit(
            trainGenerator,
            steps_per_epoch= trainGenerator.n//trainGenerator.batch_size,
            epochs=trainParams["Epochs"],
            callbacks= callbacks,
            validation_data= valGenerator,
            validation_steps=valGenerator.n // valGenerator.batch_size
        )
    
