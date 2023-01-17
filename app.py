from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import keras # machine learning model training  

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/bestmodel.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(155, 155))

    img1 = keras.utils.load_img(img_path, target_size=(155, 155)) 
    img1  = keras.utils.img_to_array(img1) 
    img1 = np.expand_dims(img1, axis=0) 
    
    result = model.predict(img1/255)

    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'UPLOADS', secure_filename( f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        acc = preds[0][0]*100
        # Process your result for human
        if   preds[0][1] < 0.5:  # la barre minimal pour la prediction p ou n 
            return f"Organic avec precision {acc:.2f}%" 
        else:
            return f"Recyclable avec precision {acc:.2f}%"
   
    return None


if __name__ == '__main__':
    app.run(debug=True)

