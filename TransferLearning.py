#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:35:12 2019

@author: davidfelipealvear

"""


# Libraries 
import tensorflow as tf
from deep_tools import deep_tools

print("TensorFlow version is ", tf.__version__)

"""
Define variables
"""
## Defined variables
path_dataset = "Dataset"
image_size = 75 # All images will be resized to 40x40x3
IMG_SHAPE = (image_size, image_size, 3) ## structure of the image classifier
batch_size = 32
no_classes = 2
mode = 'binary' # For train generators
## Check_point name and model save name
name_model = "3VGG19"
checkpoint = "MC1"+name_model+".h5"
model_save = "M"+name_model+".h5"

"""
Resume the information in the defined dataset
"""
## Resume the dataset information
dep_tools = deep_tools("Dataset")
dep_tools.resume_dataset(3)

"""
Create image generators
"""
## Image data generators
train_generator, validation_generator, test_generator = dep_tools.image_data_generator(
        mode, image_size, batch_size)

"""
Model creation
"""
## Create the base model from the pre-trained model MobileNet V2
model, baseModel = dep_tools.model_creation(IMG_SHAPE, num_classes=no_classes, verbose=True)

"""
Checkpoint 
"""
## Create variables to store callbacks checkpoint and early stopping
callback_list = dep_tools.callbacks(checkpoint)

"""
Transfer Learning
"""
## Freeze layers of neural network transfer learning
dep_tools.setup_to_transfer_learn(model,baseModel)

"""
Training neural network - TransferLearning
"""
epochs = 15
## train the model defined
history = dep_tools.train_neural_network(model, epochs,
                                          train_generator,
                                          validation_generator,
                                          batch_size,
                                          callbacks=callback_list)

"""
Fine tunning
"""
freeze_layers = 140
dep_tools.setup_to_finetune(model, freeze_layers)

"""
Graph the trainig process
"""
dep_tools.graph_training_info(history)

"""
Training neural network - Fine Tunning
"""
epochs = 2
## train the model defined
history = dep_tools.train_neural_network(model, epochs,
                                          train_generator,
                                          validation_generator,
                                          batch_size,
                                          callbacks=callback_list)

"""
Graph the trainig process
"""
dep_tools.graph_training_info(history)

"""
Perform test analysis
"""
results = dep_tools.test_analysis(model, test_generator, no_classes)

"""
Save the weights and model trained
"""
dep_tools.model_save(model, model_save, weights=False)






