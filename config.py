"""config.py
"""

import os
import types
import logging
import shutil
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.show_sample_images = config_arg['image']['show_image']
        self.imageFolderPath = config_arg['image']["raw_data"]['path']
        print("platform", self.platform=="gcp")
        if self.platform == 'gcp':
            self.TRAIN_DIR = str(config_arg['data']['gcp']['train'])
            self.EVAL_DIR = str(config_arg['data']['gcp']['test'])
            self.destination = config_arg['save_model']['gcp']['path_prefix']
            ############################################################################################
            #                            Path Varaibles for GCP or GS Bucket                           #
            ############################################################################################
            self.checkpoint_path = None
            self.log_path = None
            self.dirpath = os.getcwd()
            print("current directory is : " + self.dirpath)

            self.save_model_path = self.destination + "save_model/" + str(self.model_version)  
            
            self.destination = self.destination + self.model_version
            if os.path.isdir(self.destination):
                shutil.rmtree(self.destination, ignore_errors = True)
                logger.info("Removed old {}".format(self.destination))
            self.checkpoint_path = self.destination + "/checkpoints"
            self.log_path = self.destination + "/logs"
            logger.info("Finished creating {}".format(self.destination))
        else:
            self.TRAIN_DIR = config_arg['data']['local']['train']
            self.EVAL_DIR = config_arg['data']['local']['test']
            self.destination = config_arg['save_model']['local']['path_prefix']
            ############################################################################################
            #                            Path Varaibles For Local                                      #
            ############################################################################################
            self.checkpoint_path = None
            self.log_path = None
            self.dirpath = os.getcwd()
            print("current directory is : " + self.dirpath)

            self.save_model_path = self.destination + "save_model/" + str(self.model_version)  
            self.destination = self.destination + self.model_version
            if os.path.isdir(self.destination):
                shutil.rmtree(self.destination, ignore_errors = True)
                logger.info("Removed old {}".format(self.destination))
            os.makedirs(self.destination)
            self.checkpoint_path = self.destination + "/checkpoints"
            os.mkdir(self.checkpoint_path)
            self.log_path = self.destination + "/logs"
            os.mkdir(self.log_path)
            logger.info("Finished creating {}".format(self.destination))





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
  

    