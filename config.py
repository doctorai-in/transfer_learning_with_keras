"""config.py
"""

import os
import types


config = types.SimpleNamespace()

# Subdirectory name for saving trained weights and models
config.SAVE_DIR = 'saves'

# Subdirectory name for saving TensorBoard log files
config.LOG_DIR = 'logs'


# Default path to the ImageNet TFRecords dataset files
config.DEFAULT_DATASET_DIR = os.path.join(
    os.environ['HOME'], '/lab/GCP/Transfer_Learning/tfrecord')

# Number of parallel works for generating training/validation data
config.NUM_DATA_WORKERS = 8

# Do image data augmentation or not
config.DATA_AUGMENTATION = True

class variable_config():

    def __init__(self, config_arg):   
        ######### Load config ######################################
        ######## set argument ######################################
        self.epochs = int(config_arg['training']['epoch'])
        print('epochs == {}'.format(self.epochs))
        self.batch_size = int(config_arg['training']['train_batch'])
        self.val_batch_size = int(config_arg['training']['val_batch'])
        self.model_version = str(config_arg['training']['version'])
        self.loss = config_arg['training']['loss'] # binary_crossentropy or categorical_crossentropy
        self.class_mode = config_arg['data_generator']['class_mode'] # binary or categorical
        self.img_target_size = config_arg['image']['size']


        self.MODEL_ARCHITECTURE = config_arg['type'] + self.model_version + ".yaml"
        self.CHECK_POINT = config_arg['type'] + self.model_version + '_weights_{epoch:02d}-{loss:.2f}'
        self.MODEL_FILE = config_arg['type'] + self.model_version + '_model.h5'
        self.WEIGHT_FILE = config_arg['type'] + self.model_version + '_weights_.h5'
        self.HISTORY_FILE = 'history_' + config_arg['type'] + self.model_version + '.csv'
        self.LR_FILE='lr_' + config_arg['type'] + self.model_version + '.csv'
        self.platform = str(config_arg['platform'])
        print("platform", self.platform=="gcp")
        if self.platform == 'gcp':
            self.TRAIN_DIR = str(config_arg['data']['gcp']['train'])
            self.EVAL_DIR = str(config_arg['data']['gcp']['test'])
            self.destination = config_arg['save_model']['gcp']['path_prefix']
        else:
            self.TRAIN_DIR = config_arg['data']['local']['train']
            self.EVAL_DIR = config_arg['data']['local']['test']
            self.destination = config_arg['save_model']['local']['path_prefix']
