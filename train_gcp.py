###########################################################################
#                            Copyright                                    #
###########################################################################

from __future__ import division
import tensorflow as tf
import logging
import argparse
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam
import string
import os
import pandas as pd
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import load_model
import shutil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, add, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml
import tensorflow.keras as keras
import sys
from load_tf_record import TFRecordLoader
from utils.dataset import get_dataset
from tensorflow.keras.models import save_model
import datetime
from config import variable_config
#############################################################################
# 
#############################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########### Training data generator ########

stream = open('transfer_learning_with_keras/config.yaml', 'r')
config_arg = yaml.safe_load(stream)
config = variable_config(config_arg)
#data_loader = TFRecordLoader(batch_size, img_target_size)
print(config.TRAIN_DIR)
print(config.EVAL_DIR)
train_generator = get_dataset(config.TRAIN_DIR, 'train', batch_size=config.batch_size)
eval_generator =  get_dataset(config.EVAL_DIR, 'test', batch_size=1)
train_generator.take(1)
print("train_generator: ", train_generator)
debug=False
if debug:
    image_batch, label_batch = next(iter(train_generator))
    data_loader.show_batch(image_batch.numpy(), label_batch.numpy())
    print("train_generator: ", train_generator)
############ Define Model ##############
nb_epochs = config.epochs


# Create base model
basemodel = keras.applications.Xception(
    weights='imagenet',
    input_shape=(config.img_target_size, config.img_target_size, 3),
    include_top=False)

# Freeze base model
basemodel.trainable = False

# Create new model on top.
inputs = keras.Input(shape=(config.img_target_size, config.img_target_size, 3))
x = basemodel(inputs, training=False)

# Full connection
x = keras.layers.GlobalAveragePooling2D()(x) 
fc2_out = Dense(1)(x)

model = tf.keras.Model(inputs, fc2_out)

#for layer in basemodel.layers:
    #layer.trainable = False


opt = Adam(lr=5e-4, decay=0.1)
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )

logger.debug("Model summary...")
model.count_params()
model.summary()



############# Support infra #########
checkpoint_path = None
log_path = None
destination = config.destination
dirpath = os.getcwd()
print("current directory is : " + dirpath)
save_model_path = destination + "save_model/" + str(config_arg['training']['version'])  
destination = destination + config.model_version
if os.path.isdir(destination):
    shutil.rmtree(destination, ignore_errors = True)
    logger.info("Removed old {}".format(destination))
os.makedirs(destination)
checkpoint_path = destination + "/checkpoints"
os.mkdir(checkpoint_path)
log_path = destination + "/logs"
os.mkdir(log_path)
logger.info("Finished creating {}".format(destination))


# Write model architecture
# serialize model to JSON ot YAML eg. model.to_json() or model.to_yaml()
# loding from json and yaml the architecture : 
# load YAML and model
'''
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''
model_json = model.to_yaml()
with open(os.path.join(destination , config.MODEL_ARCHITECTURE), "w") as json_file:
    json_file.write(model_json)

run_id = "cat_dog-" + str(config.batch_size) + "-" + '' \
.join(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))


from keras.callbacks import LearningRateScheduler

def schedule(epoch, lr):
    if epoch%25 == 0:
        return lr * 0.5
    return lr
#lr_scheduler = LearningRateScheduler(schedule, verbose=0)

callbacks = [
    ModelCheckpoint(
        os.path.join(checkpoint_path, config.CHECK_POINT),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto"
        ),
    TensorBoard(log_dir= os.path.join(log_path,run_id) ),
    #lr_scheduler,
    ]

########### Run the model #############
hist = model.fit(
train_generator,
callbacks = callbacks,
batch_size=16,
steps_per_epoch = 10,# there are around 9776 images, % by batch size of 16
epochs=nb_epochs,
validation_data=eval_generator,
shuffle=True,
validation_steps=2,
#verbose=2,
)

## Print lr ###
#print(lr_scheduler.history)


    

######## Save Model ###############
# refer to article : https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#https://github.com/damienpontifex/mobilenet-classifier-transfer/blob/master/binary_classifier_train.py
#https://www.tensorflow.org/tfx/serving/api_rest
#https://www.tensorflow.org/tfx/serving/api_rest
#https://www.tensorflow.org/guide/saved_model#exporting_custom_models
#https://www.tensorflow.org/guide/saved_model
save_path_pb = save_model_path
logger.info("save model")
#model.save(os.path.join(destination, MODEL_FILE), save_format="tf")

#model.save(save_path_pb, save_format="tf", signatures=serving)
tf.saved_model.save(model, save_path_pb)
model.save_weights(os.path.join(destination, config.WEIGHT_FILE), overwrite=True)
pd.DataFrame(hist.history).to_csv(os.path.join(destination, config.HISTORY_FILE))
###################################################################################
#                                 END OF code                                     #
###################################################################################







###################################################################################
#                                 Code Snippet                                    #
###################################################################################
#FILENAMES_TRAIN = tf.io.gfile.glob("/home/omen/lab/GCP/Transfer_Learning/tfrecord/train*")
#FILENAMES_EVAL = tf.io.gfile.glob("/home/omen/lab/GCP/Transfer_Learning/tfrecord/test*")
'''
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def serving(input_image):

    # Convert bytes of jpeg input to float32 tensor for model
    def _input_to_feature(img):
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, 224, 224)
        return img
    img = tf.map_fn(_input_to_feature, input_image, dtype=tf.float32)

    # Predict
    predictions = model(img)

    

    # Single output for model so collapse final axis for vector output
    predictions = tf.squeeze(predictions, axis=-1)

    
    return {
        'probabilities': 0
    }'''