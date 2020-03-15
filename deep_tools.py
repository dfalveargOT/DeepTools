#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 14 12:25:14 2019
@author: davidfelipe
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

class deep_tools:
    
    def __init__(self, base_path):
        
        print(" ")
        print("Deep Learning training tools")
        print("This tool will manage the training dataset for neural networks")
        print("Suggested the order of the dataset in [trainig, validate, test] folders")
        print(" ")
        self.base_dir = base_path
        self.train_dir = []
        self.validation_dir = []
        self.test_dir = []
        self.model_path = 'Model'
        flag = os.path.isdir(base_path)
        if flag== False:
            print("Path given is not a folder")
        self.font = {'family': 'serif',
                    'color':  'black',
                    'weight': 'normal',
                    'size': 13}
        # Config GPU for use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.keras.backend.set_session(tf.Session(config=config))
            
    def model_creation(self, img_shape, num_classes = 2,
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'],
                   verbose = False,
                   compilation = False):
    
        """
        model_creation - Function to load neural network for transfer learning 
        
        input:
            - IMG_SHAPE : Shape of the images to fit into the deep network
            - tune_layers : number of layer to freeze from zero tending to the top layers
            - loss : Define loss function, default 'sparse_categorical_crossentropy'
            - metrics : Define the metrics to compile the model default ['accuracy']
            - verbose : Flag to activate logs information about the loaded model
            - compilation : Flag if compile the model
        
        output:
            - model : model compiled ready to fit and train
        
        Observations :
            - Modify the function to fit the conditions for the application needs
        
        """
        
        base_model = tf.keras.applications.DenseNet201(input_shape=img_shape, weights='imagenet', include_top=False)
        base_model.trainable = False
      
        ####### Start the modification of the top layers
        
        model = tf.keras.Sequential([
          base_model,
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dense(1024, activation='sigmoid'), ## Fully connected layer 1024
          keras.layers.Dropout(0.5),
          #keras.layers.Dense(512, activation='tanh'), ## Fully connected layer 1024
          #keras.layers.Dropout(0.3),
          #keras.layers.Dense(512, activation='sigmoid'), ## Fully connected layer 1024
          #keras.layers.Dense(256, activation='relu'), ## Fully connected layer 1024
          keras.layers.Dense(num_classes, activation=tf.nn.softmax)])
        
        ####### Finish the modification of the top layers
        
        ## Compilation
        if compilation:
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=loss,
                        metrics=metrics)
        
        if verbose:
            model.summary()
            print("Number of layers in the base model : ", len(base_model.layers))
            print("Number of trainable variables : " + str(len(model.trainable_variables)))
        
        return model, base_model
    

    def setup_to_transfer_learn(self, model, base_model):
        """
        setup_to_transfer_learn - Function to configure transfer learning

        Input:
            - model : Model created with top layer
            - base_model : CNN basis model for transfer learning
        """

        """Freeze all layers and compile the model"""
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    

    def setup_to_finetune(self, model, layers_Freeze):
        """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
        Args:
            model: keras model
        """
        for layer in model.layers[:layers_Freeze]:
            layer.trainable = False
        for layer in model.layers[layers_Freeze:]:
            layer.trainable = True
        #model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_neural_network(self, model, epochs, 
                             train_generator, 
                             validation_generator,
                             batch_size, callbacks, 
                             workers=4):
        """
        train_neural_network - Function to perform training of model defined
        
        input :
            - model : Compiled model to fit with generator
            - train_generator : Generator created in base of train dataset
            - validation_generator : Generator created in base of validation dataset
            - batch_size : size of packets to be divided the dataset train images
            - callbacks : callbacks list to add in the train process
            - workers : create parallelism for training
            
        output :
            - history : resume of the complete training
            
        """
        
        # Define steps for training in base on generators
        steps_per_epoch = train_generator.n // batch_size
        validation_steps_ = validation_generator.n // batch_size
        
        ## Use fit generator function from model to train the neural network
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = steps_per_epoch,
                                      epochs=epochs,
                                      workers=workers,
                                      validation_data=validation_generator,
                                      validation_steps=validation_steps_,
                                      class_weight='auto',
                                      callbacks=callbacks)
        return history
    
    def callbacks(self, checkpoint_name="MDefault.h5", path_save= './',monitor='val_loss', verbose=0,
                  mode_checkpoint='min', patience_earlyStop=35, baseline_ES=0.08,
                  tensorboard=True, earlystop=True, checkpoint=True):
        
        """
        callbacks - Function for create callbacks checkpoint and earlystop
        
        input :
            - checkpoint : path + name for save the checkpoint
            - monitor : Variable that will monitor the callback's functions
            - verbose : default 1, flag to see logs information
            - mode_checkpoint : Type checkpoint configuration default 'min'
            - patience_earlyStop : Number of epochs to leave pass
            - baseline_ES : basline difference to stop the training early stop
            - tensorboard : Flag to forward the tensorboard callback
            - earlystop : Flag to forward the earlystop callback
            - checkpoint : Flag to forward the checkpoint callback
                
        output :
            - callback_list : return list of callback functions
        """
        checkpoint_path = os.path.join(path_save, checkpoint_name)
        modelCheckpoint = ModelCheckpoint(filepath=checkpoint_path, monitor=monitor, mode=mode_checkpoint, verbose=verbose, save_best_only=True)
        earlyStop = EarlyStopping(monitor=monitor, verbose=verbose, patience=patience_earlyStop, baseline=baseline_ES)
        
        if tensorboard:
            logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        if checkpoint and earlystop==False and tensorboard==False:
            callback_list = [modelCheckpoint]
        elif earlyStop and checkpoint==False and tensorboard==False:
            callback_list = [earlyStop]
        elif earlyStop==False and checkpoint==False and tensorboard:
            callback_list = [tensorboard_callback]
        elif checkpoint and earlystop and tensorboard==False:
            callback_list = [modelCheckpoint,earlyStop]
        elif checkpoint and earlystop and tensorboard:
            callback_list = [modelCheckpoint,earlyStop,tensorboard]

            
        
        return callback_list
    
    def model_save(self,model, name, weights=False, path_save='./'):
        """
        model_save - Function to perform the save of trained model
        
        input :
            - model : model checked and trained
            - name : name to save the model and/or weights
            - weights : flag to control the save of 
        
        Output :
            - model saved in the dir ./Model
        """
        files = os.listdir(path_save)
        flag = False
        path = os.path.join(path_save, self.model_path)
        for file in files:
            if file==self.model_path:
                flag = True
                break
        if flag == False:
            os.mkdir(path)
            print("Creating the path to save the model")
        # Flag to control the weights save
        if weights:
            model.save_weights(path + "Weights" + name)
        model.save(path + name)
        print("model save function procedure completed")

    def save_model_json(self, model, name_model):
        model_json = model.to_json()
        with open(name_model + '.json', "w+") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights('Weights' + name_model + '.h5')
        print("Saved model to disk")
        

    def load_model_json(self, path_json, path_weights):
        json_file = open(path_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tf.keras.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return loaded_model
    
    def graph_training_info(self, history):
        """
        graph_training_info - Function for create a resume graph of the train process
        
        input :
            history : Resume given by the model training
            
        output :
            graph : Graphicof the training resume process
        """

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')
        
        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,max(plt.ylim())])
        plt.title('Training and Validation Loss')
        plt.show()
        
    def test_analysis(self, model, test_generator, classes, class_names):
        """
        test_analysis - Function to carry out a performance analysis of the network
        
        input : 
            - model : model compiled trained and loaded
            - test_generator : generator of the test dataset
        
        output :
            - log information about the predicted information in the test dataset
        """
        ## Evaluate generator
        score = model.evaluate_generator(test_generator, 3)
        print("Global results for test : " + str(score))
        
        ## Predict generator
        results = model.predict_generator(test_generator)

        y_pred = np.argmax(results, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(test_generator.classes, y_pred))
        print('Classification Report')
        #class_names = ['Piña', 'Suelo']
        print(classification_report(test_generator.classes, y_pred, target_names=class_names))

        return results

    def prepare_list(self, list_images, size = 200):
        """
        prepare_list - Function to divide the dataset into batches with given size
        
        Input:
            list_images - list of the dataset
            size - size of the batches
        
        Return:
            list_batches - list of the dataset divided in batches
        """
        batch = []
        list_batches = []
        counter = 0
        for item, img in enumerate(list_images):
            batch.append(img)
            counter += 1
            if counter >= size:
                counter = 0
                list_batches.append(batch)
                batch = []
            elif item == len(list_images) - 1:
                list_batches.append(batch)
        return list_batches
        

    def test_graph(self, model, image_size, directory, score_min, size = 200):
        """
        test_graph - Function for perform a graphic analysis of model performance

        Input :
            - model : Compiled keras model ready for predict
            - image_size : Size of the image input of the model
            - directory : Directory where the dataset is placed
            - score_min : Score Threshold [c1, c2, c3]
            - size : Batch size of the prediction model process

        Output :
            - 2DGraph : Plot the results of the prediction in a graph
        """
        results_classes = [] # Store for each class list of batches [[[batch],[batch]], [[batch],[batch]]]
        # Generate the paths for each class in the given diractory
        for item in self.classes:
            path = os.path.join(directory, item)
            elements = os.listdir(path)
            batch_elements = self.prepare_list(elements, size) # create list of batches for process later
            # Load each batch and predict it
            results_class = self.predict_batches(model, path, batch_elements, image_size) 
            results_classes.append(results_class)
        self.results_resume(results_classes, score_min)
        self.graph2D(results_classes)
    
    def results_resume(self, results_list, score_min):
        """
        results_resume - Function that provide an threshold analysis 

        Input : 
            - results_list : Results list containing each class trained
            - score_min : Score Threshold [c1, c2, c3]
        
        Output :
            - log information of the count in base of the threshold value
        """
        for idx, class_result in enumerate(results_list):
            shape = class_result.shape
            print(shape)
            for i in range(0,shape[1]):
                axis = class_result[:,i]
                axis_bool = np.where(axis > score_min[i],0,1)
                results_count = np.count_nonzero(axis_bool, axis=0)
                print("For class : " + str(self.classes[idx]) + " Count score : " + str(i) + " Results : " + str(results_count))
                print("Total elementes : " + str(len(axis)) + "\n")
    
    def predict_batches(self, model, base_path, list_batches, image_size):
        """
        Predict_batches - Funntion to predict multiple batches

        Input:
            model - Compiled keras model ready for predictions
            base_path - Path base for the batch list given
            list_batches - batch list of divided dataset
            image_size - Image size for modify the image input

        Output:
            results_batches - np.array of the results of the prediction
        """
        for item, batch in enumerate(list_batches):
            batch_imgs = self.load_batch(base_path, batch)
            result = self.predict_batch(model, batch_imgs, image_size)
            if item == 0:
                results_batches = result.copy()
            else:
                results_batches = np.vstack((results_batches, result))
        return results_batches

    def predict_batch(self, model, list_img,image_size):
        """
        predict_batch - Funntion to predict list of images

        Input:
            model - Compiled Keras model 
            list_img - list of string names for images

        Output:
            results - np.array of the results of the prediction
        """
        batch_size = len(list_img)
        batch_holder = np.zeros((batch_size, image_size, image_size, 3))
        for j, img in enumerate(list_img):
            temp = cv2.resize(img.astype("uint8"),(image_size, image_size))
            temp = image.img_to_array(temp)
            temp = np.expand_dims(temp, axis=0)
            temp = keras.applications.mobilenet.preprocess_input(temp)
            batch_holder[j, :] = temp[0]
        results = model.predict_on_batch(batch_holder)
        batch_holder = []
        return results

    def load_batch(self, base_path, batch):
        """
        load_batch - Function for load batch of images in a given directory
        
        Input :
            - base_path : Path from the top project directory to the container directory
            - batch : String list of the images in the container directory
        
        Return :
            - img_batch : Opencv batch images loaded in the software
        """
        img_batch = []
        for image_name in batch:
            path_image = os.path.join(base_path, image_name)
            img_cv = cv2.imread(path_image)
            img_batch.append(img_cv)

        return img_batch


    def evaluate_list(self, model,folder,classes=2,num_events=1, imageSize=75):
        """
        evaluate_list : Function to perform random analysis in a dataset showing information
        Input:
            - model : compiled weighted model ready for predictions
            - folder : training, testing or validation folder containing the classes
            - classes : num of classes contained
            - num_events : number of predictions/windows to be done
            - imageSize : size of the image to be loaded and resized
        """
        listImages = self.images_dataset(folder, classes) 
        for i in range(0,num_events):
            images = []
            predictions = []
            print("Event number : " + str(i))
            # Extract a image sample from the directory for each class
            for idx,lista in enumerate(listImages):
                total = len(lista)
                imageName = lista[np.random.randint(total)]
                path = os.path.join(folder, self.classes[idx], imageName)
                Image = cv2.imread(path)
                print(path)
                print(Image.shape)
                images.append(Image)
            # Predict for the extracted samples
            for Image in images:
                try:
                    predict_image = self.preprocessing_image(Image, imageSize)
                    results = model.predict(predict_image)
                    predictions.append(results)
                except:
                    continue
            # Graph the results ********************
            self.graph_results2(images, predictions)
            

    def graph_results2(self, images, predictions):
        """
        graph_results2 - Function to show image and predicted values for 2 images
        Input:
            - images : List of images to show (two only)
            - predictions : Results array of deep learning predictions
        Output:
            - Graph : Image with the desired information
        """
        results_prediction = []
        # Get prediction information format
        for predict in predictions:
            tittle = ("Planta: "+ str(round(predict[0][0],3)) + "  No_Planta : " + str(round(predict[0][1],3)))#+ "  No_Planta2 : " + str(round(predict[0][2],3)))
            results_prediction.append(tittle)
        # Graph the results
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.tight_layout() 
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_RGBA2RGB))
        plt.title(results_prediction[0], fontdict=self.font)
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(images[1], cv2.COLOR_RGBA2RGB))
        plt.title(results_prediction[1], fontdict=self.font)
        plt.show()
        
            
    def preprocessing_image(self,ImageProcessing, imageSize):
        """
        preprocessing_image - Function from keras to preprocess image for prediction
        Input:
            image - numpy image 
            imageSize - size of one dimension of the image for resize
        output:
            image_cv - Stack numpy array containing image information
        """
        image_cv = cv2.resize(ImageProcessing,(imageSize,imageSize))
        image_cv = image.img_to_array(image_cv)
        img_array_expanded_dims = np.expand_dims(image_cv, axis=0)
        image_cv = keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
        return image_cv

    def images_dataset(self, folder, classes):
        """
        images_dataset - Function for extract the items in defined folder
        
        Input : 
            - folder : Folder base to get the class information
            - classes : number of folder in the base folder 

        Output :
            - listImages : List of arrays corresponding to the items in each class
        """
        listImages = []
        for i in range(0,classes):
                directory = os.path.join(folder, self.classes[i])
                listClass = os.listdir(directory)
                listImages.append(listClass)
        return listImages
        
    def graph2D(self, results_list):
        """
        graph2D - Function to generate 2D graph of the results
        Input:
            - results_list : List of the predicted dataset in the model
        Output:
            - 2D matplotlib graph 
        """
        x = []
        y = []
        color = ['b', 'r', 'g']
        marker = ['v', 'd', 'o']
        # Extract points for plot
        for lista in results_list:
            x.append(lista[:,0])
            y.append(lista[:,1])

        # Create figure 
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for idx in range(0,len(x)):
            ax.scatter(x[idx], y[idx],marker=marker[idx], c=color[idx])
        
        plt.show()

    def resume_dataset(self, classes):
        
        """
        resume_dataset - Function to perform a review of the dataset
        
        Input:
            - classes : number of classes defined in the dataset
            
        Observation: 
            - For training a neural network is necessary to have dataset
            for training, velidation and testing, each below folder contain the same 
            structure naming inside the classes divided in folders
        """
        
        # List to store the paths in the dataset
        self.train_dirs = []
        self.validate_dirs = []
        self.test_dirs =[]
        self.classes = []
        # Define the classes
        self.train_dir = os.path.join(self.base_dir, 'training')
        self.validation_dir = os.path.join(self.base_dir, 'validate')
        self.test_dir = os.path.join(self.base_dir, 'test')
        elements = os.listdir(self.train_dir)
        print(self.train_dir)
        counter = 0
        ## Find the classes names
        for item in elements:
            path_element = os.path.join(self.train_dir, item)
            flag = os.path.isdir(path_element)
            if item == "__pycache__":
                continue
            if flag:
                self.classes.append(item)
                counter += 1
        
        if counter != classes:
            print("Don't match the quantity of classes defined with founded")
        
        else:
            ## List and count the elements in the folders
            ## Training
            for i in range(0,classes):
                directory = os.path.join(self.train_dir, self.classes[i])
                print ('Total training '+ self.classes[i] + ' images:', 
                       len(os.listdir(directory)))
                self.train_dirs.append(directory)
            ## Validation
            for i in range(0,classes):
                directory = os.path.join(self.validation_dir, self.classes[i])
                print ('Total validation '+ self.classes[i] + ' images:', 
                       len(os.listdir(directory)))
                self.validate_dirs.append(directory)
            ## Testing
            for i in range(0,classes):
                directory = os.path.join(self.test_dir, self.classes[i])
                print ('Total test '+ self.classes[i] + ' images:', 
                       len(os.listdir(directory)))
                self.test_dirs.append(directory)
            print(" ")
            
    def image_data_generator(self, mode='binary', image_size=200, batch_size=32, flip_Augment=True, rot_augment=90):
        """
        image_data_generator - Function to create ImageDataGenerators
                                using keras to load the dataset
        
        Input :
            - mode : Configuration keras data generator
            - image_size : Size of the images to be loaded
            - batch_size : Size of image packets to be done
            - flip_Augment : Data Augmentation Flip vertical/horizontal 
            - rot:augment : Data Augmentation rotation angle 

        Output :
            train_generator - Generator flow from train directory
            validate_generator - Generator flow from validate directory
            test_generator - Generator flow from test directory
        """
        # Rescale all images by 1./255 and apply image augmentation/reduction
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255, vertical_flip=flip_Augment, rotation_range=rot_augment)

        validation_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255, vertical_flip=flip_Augment,rotation_range=rot_augment)

        test_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255, vertical_flip=flip_Augment, rotation_range=rot_augment)
        
        ### Load the dataset in the generators
                    # Flow training images in batches of 20 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(
                        self.train_dir,  # Source directory for the training images
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        # Since we use binary_crossentropy loss, we need binary labels
                        class_mode=mode)
        
        # Flow validation images in batches of 20 using test_datagen generator
        validation_generator = validation_datagen.flow_from_directory(
                        self.validation_dir, # Source directory for the validation images
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        class_mode=mode)
        
        test_generator = test_datagen.flow_from_directory(
                        self.test_dir, # Source directory for the validation images
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        class_mode=mode)
        
        return train_generator, validation_generator, test_generator
        
    
            