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
from LoadImageFromFolder import ImageFolder
import tensorflow_datasets as tfds
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#############################################################################
#                         Loading Configuration                             #
#############################################################################

stream = open('config.yaml', 'r')
config_arg = yaml.safe_load(stream)
config = variable_config(config_arg)

#############################################################################
#                          Loading Data                                     #
#############################################################################

def build_data(config):  
    '''
    return train_ds, test_ds
    ''' 
    builder = ImageFolder(config.imageFolderPath)
    print(builder.info) # num examples, labels... are automatically calculated
    train_ds, eval_ds = builder.as_dataset(split=['train/', 'test/'], shuffle_files=True)
    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print("Number of test samples: %d" % tf.data.experimental.cardinality(eval_ds))
    if config.show_sample_images:
        tfds.show_examples(train_ds, builder.info)
    else :
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(train_ds.take(9)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(image)
            plt.title(int(label))
            plt.axis("off")


    size = (config.img_target_size, config.img_target_size)
    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    eval_ds = eval_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    batch_size = 32

    train_ds = train_ds.cache().batch(config.batch_size).prefetch(buffer_size=10)
    eval_ds = eval_ds.cache().batch(config.val_batch_size).prefetch(buffer_size=10)

    return train_ds, eval_ds


train_ds, eval_ds = build_data(config)    

#############################################################################
#                       Define Model Architecture                           #
#############################################################################

nb_epochs = config.epochs

def create_model(config):
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
    return model


model = create_model(config)
logger.info("MODEL LOADED SUCCESSFULLY")

#############################################################################
#                       Define Data store Directory                         #
#############################################################################
checkpoint_path = None
log_path        = None
destination     = config.destination
dirpath         = os.getcwd()

print("current directory is : " + dirpath)

save_model_path  = config.save_model_path
destination      = config.destination
checkpoint_path  = config.checkpoint_path
log_path         = config.log_path
logger.info("Finished creating {}".format(destination))

##############################################################################
#                      Training Configuration                                #
##############################################################################
# Write model architecture
# serialize model to JSON ot YAML eg. model.to_json() or model.to_yaml()
# loding from json and yaml the architecture : 
# load YAML and model

#model_json = model.to_yaml()
#with open(os.path.join(destination , config.MODEL_ARCHITECTURE), "w") as json_file:
    #json_file.write(model_json)

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

#################################################################################
#                         Start Training                                        #
#################################################################################
hist = model.fit(
    train_ds,
    callbacks = callbacks,
    batch_size=16,
    steps_per_epoch = 10,# there are around 9776 images, % by batch size of 16
    epochs=nb_epochs,
    validation_data=eval_ds,
    shuffle=True,
    validation_steps=2,
    #verbose=2,
    )
#print(lr_scheduler.history)

##################################################################################
#                             Saving Model                                       #
##################################################################################
# refer to article : https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#https://github.com/damienpontifex/mobilenet-classifier-transfer/blob/master/binary_classifier_train.py
#https://www.tensorflow.org/tfx/serving/api_rest
#https://www.tensorflow.org/tfx/serving/api_rest
#https://www.tensorflow.org/guide/saved_model#exporting_custom_models
#https://www.tensorflow.org/guide/saved_model
#https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/transfer_learning.ipynb#scrollTo=dynvLAC4Vl59

save_path_pb = save_model_path
logger.info("save model")
#model.save(os.path.join(destination, MODEL_FILE), save_format="tf")

#model.save(save_path_pb, save_format="tf", signatures=serving)
tf.saved_model.save(model, save_path_pb)
#model.save_weights(os.path.join(destination, config.WEIGHT_FILE), overwrite=True)
pd.DataFrame(hist.history).to_csv(os.path.join(destination, config.HISTORY_FILE))
###################################################################################
#                                 END OF code                                     #
###################################################################################