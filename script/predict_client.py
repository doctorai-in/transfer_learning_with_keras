import urllib
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests
import numpy as np
import json


# Load image
img_path = '/home/omen/lab/GCP/Transfer_Learning/data/image/test/dogs/dog.4032.jpg'
img = load_img(img_path, target_size=(224, 224))
img = img_to_array(img)
# Create payload
payload = {"instances": [img.tolist()]}

# Make request
headers = {"content-type": "application/json"}
res = requests.post("http://localhost:8501/v1/models/save_model:predict", json=payload, headers=headers)
print('ok')
res = res.json()
print(res)

'''body = {
    'instances': [
        { 'input_2': img.tolist() }
    ]
}

data = json.dumps(body)

req = urllib.request.Request(
    'http://localhost:8501/v1/models/save_model:predict',
    data=data.encode(),
    method='POST'
)

res = urllib.request.urlopen(req)
res = res.read().decode()
print(res)'''