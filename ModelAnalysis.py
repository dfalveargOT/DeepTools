# Libraries 
import tensorflow as tf
from deep_tools import deep_tools
#print("TensorFlow version is ", tf.__version__)

## Defined variables
name_model = "3-ModelM3DenseNet169.h5"
path_model = "./HistoryModels/"
path_dataset = "Dataset"
image_size = 50 # All images will be resized to 75x75x3
IMG_SHAPE = (image_size, image_size, 3) ## structure of the image classifier
batch_size = 22
no_classes = 2
mode = 'binary'

## Resume the dataset information
dep_tools = deep_tools("Dataset")
dep_tools.resume_dataset(no_classes)

"""
Create image generators
"""
## Image data generators
train_generator, validation_generator, test_generator = dep_tools.image_data_generator(
        mode, image_size, batch_size)

# Load model
model = tf.keras.models.load_model(path_model + name_model, compile=True)

# Save model json
#dep_tools.save_model_json(model, "1-ModelMobileNet58")

# Performance evaluation of the model
print("loaded")
#resultsPrediction = dep_tools.test_analysis(model, test_generator, no_classes)
thresh = [0.9, 0.5, 1]
dep_tools.test_graph(model, image_size, dep_tools.test_dir, thresh, 200)

# Graph 2D of the results for each class
#dep_tools.graph2D(resultsPrediction)

# Evaluation the model in the test o validation dataset
#dep_tools.evaluate_list(model, dep_tools.validation_dir, 2, 5,image_size)

