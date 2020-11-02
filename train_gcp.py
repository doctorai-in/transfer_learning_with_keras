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
#sys.path.append('../')
from utils.dataset import get_dataset
########### Logger setup ##############
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

######### Load config ######################################
stream = open('transfer_learning_with_keras/config.yaml', 'r')
config_arg = yaml.safe_load(stream)
######## set argument ######################################
epochs = int(config_arg['training']['epoch'])
print('epochs == {}'.format(epochs))
batch_size = int(config_arg['training']['train_batch'])
val_batch_size = int(config_arg['training']['val_batch'])
model_version = str(config_arg['training']['version'])
loss = config_arg['training']['loss'] # binary_crossentropy or categorical_crossentropy
class_mode = config_arg['data_generator']['class_mode'] # binary or categorical
img_target_size = config_arg['image']['size']


MODEL_ARCHITECTURE = config_arg['type'] + model_version + ".json"
CHECK_POINT = config_arg['type'] + model_version + '_weights.{epoch:02d}-{loss:.2f}.hdf5'
MODEL_FILE = config_arg['type'] + model_version + '_model.h5'
WEIGHT_FILE = config_arg['type'] + model_version + '_weights_.h5'
HISTORY_FILE = 'history_' + config_arg['type'] + model_version + '.csv'
LR_FILE='lr_' + config_arg['type'] + model_version + '.csv'
platform = str(config_arg['platform'])
print("platform", platform=="gcp")
if platform == 'gcp':
    TRAIN_DIR = str(config_arg['data']['gcp']['train'])
    EVAL_DIR = str(config_arg['data']['gcp']['test'])
    destination = config_arg['save_model']['local']['path_prefix']
else:
    TRAIN_DIR = config_arg['data']['local']['train']
    EVAL_DIR = config_arg['data']['local']['test']
    destination = config_arg['save_model']['local']['path_prefix']




########### Training data generator ########
#FILENAMES_TRAIN = tf.io.gfile.glob("/home/omen/lab/GCP/Transfer_Learning/tfrecord/train*")
#FILENAMES_EVAL = tf.io.gfile.glob("/home/omen/lab/GCP/Transfer_Learning/tfrecord/test*")

#data_loader = TFRecordLoader(batch_size, img_target_size)
print(TRAIN_DIR)
print(EVAL_DIR)
train_generator = get_dataset(TRAIN_DIR, 'train', batch_size=batch_size)
eval_generator =  get_dataset(EVAL_DIR, 'test', batch_size=1)
debug=False
if debug:
    image_batch, label_batch = next(iter(train_generator))
    data_loader.show_batch(image_batch.numpy(), label_batch.numpy())
    print("train_generator: ", train_generator)
############ Define Model ##############
nb_epochs = epochs


# Create base model
basemodel = keras.applications.Xception(
    weights='imagenet',
    input_shape=(img_target_size, img_target_size, 3),
    include_top=False)

# Freeze base model
basemodel.trainable = False

# Create new model on top.
inputs = keras.Input(shape=(img_target_size, img_target_size, 3))
x = basemodel(inputs, training=False)

# Full connection
flatten = Flatten()(x) 
fc1 = Dense(units = 128, activation = 'relu')(flatten)
fc2_out = Dense(units = 1, activation = 'sigmoid')(fc1)

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
        loss="binary_crossentropy",
        metrics=['accuracy'],
    )

logger.debug("Model summary...")
model.count_params()
model.summary()



############# Support infra #########
checkpoint_path = None
log_path = None

dirpath = os.getcwd()
print("current directory is : " + dirpath)
destination = destination + model_version
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
with open(os.path.join(destination , MODEL_ARCHITECTURE), "w") as f:
        f.write(model.to_json())

run_id = "cat_dog - " + str(batch_size) + " " + '' \
.join(random
      .SystemRandom()
      .choice(string.ascii_uppercase) for _ in range(10)
)


from keras.callbacks import LearningRateScheduler

def schedule(epoch, lr):
    if epoch%25 == 0:
        return lr * 0.5
    return lr
#lr_scheduler = LearningRateScheduler(schedule, verbose=0)

callbacks = [
    ModelCheckpoint(
        os.path.join(checkpoint_path, CHECK_POINT),
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
batch_size=16,# there are around 9776 images, % by batch size of 16
epochs=nb_epochs,
validation_data=eval_generator,
shuffle=True,
validation_steps=2,
#verbose=2,
)

## Print lr ###
#print(lr_scheduler.history)

######## Save Model ###############
logger.info("save model")
model.save(os.path.join(destination, MODEL_FILE))
model.save_weights(os.path.join(destination, WEIGHT_FILE), overwrite=True)
pd.DataFrame(hist.history).to_csv(os.path.join(destination,HISTORY_FILE))
