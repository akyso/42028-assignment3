# Usage: python app.py
import os
import time
import uuid
import json
import io
from io import BytesIO
import base64

import requests
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import SharedDataMiddleware, secure_filename
import urllib.request
from urllib.request import Request, urlopen

import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array, load_img

from predict.predict import predict


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_URL'] = '../uploads/template.jpg'

@app.route("/")
def template_test():
    return render_template('index.html', imagesource='', question='', anwer='')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            #print(file_path)
            app.config['IMAGE_URL'] = file_path

            return render_template('index.html', imagesource=app.config['IMAGE_URL'], question='', anwer='')


@app.route('/ask_question', methods=['POST'])
def my_form_post():
    text = request.form['textbox']
    question = text.capitalize()
    #print(question)

    preds = predict(app.config['IMAGE_URL'], question)

    """
    API_ENDPOINT = "http://localhost:9000/v1/models/ImageClassifier:predict"

    b64_image = ""
    # Encoding the JPG,PNG,etc. image to base64 format
    with open(file_path, "rb") as imageFile:
        b64_image = base64.b64encode(imageFile.read())

    # data to be sent to api
    data = {'b64': b64_image}

    # sending post request and saving response as response object
    r = requests.post(url=API_ENDPOINT, data=data)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))
    print(pred)
    """

    #print(preds)
    label = preds#[0][0]
    return render_template('index.html', imagesource=app.config['IMAGE_URL'], question=question, answer=label)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOAD_FOLDER']
})

"""
def encode_b64(image_path):
    b64_image = ""

    # Encoding the JPG,PNG,etc. image to base64 format
    with open(image_path, "rb") as imageFile:
        b64_image = base64.b64encode(imageFile.read())

    return b64_image


def pre_process_b64_image(file_path):
    img = img_to_array(load_img(BytesIO(base64.b64decode(file_path)), target_size=(224, 224))) / 255.

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')

    return img


def decode_inception_pred(pred):
    resp = inception_v3.decode_predictions(np.array(pred['predictions']))[0]
    result = []
    for r in resp:
        result.append({"class_name": r[1], "score": float(r[2])})

    #print(result)
    return result


def tf_predict(img):
    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    return pred

def tf_predict_data(input_string):
    instance = [{"b64": input_string}]
    data = json.dumps({"instances": instance})

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', data=data)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    return pred

@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    img = pre_process_b64_image()

    # Predict class
    pred = tf_predict(img)

    # Returning JSON response to the frontend
    return jsonify(decode_inception_pred(pred))
"""

if __name__ == "__main__":
    app.debug = False
    app.run(host='0.0.0.0', port=3000)