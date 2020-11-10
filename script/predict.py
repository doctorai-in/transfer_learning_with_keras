import urllib
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests
import numpy as np
import json
from keras.models import load_model

image = tf.keras.preprocessing.image.load_img('/home/omen/lab/GCP/Transfer_Learning/data/image/test/dogs/dog.4032.jpg', target_size=(224, 224, 3))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
model = tf.saved_model.load("/home/omen/7")
predictions = model(input_arr)
'''# Load image
img_path = '/home/omen/lab/GCP/Transfer_Learning/data/test/dogs/dog.4032.jpg'
img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)



predictions = model(img)'''

print(predictions)