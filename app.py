import os
import base64
import numpy as np
import pickle
import requests
from flask import Flask, render_template, request, send_from_directory
from werkzeug import SharedDataMiddleware, secure_filename
import tensorflow as tf
import tensorflow.keras as keras
import keras
from keras.preprocessing.sequence import pad_sequences

from settings import ANSWERS_IDX, MODEL_FOLDER, MODEL_FILE, IMG_MODEL_FILE, TOKENIZER_FILE
from Archives.predict.preprocess_image import resize_image, get_image_from_url


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(img_url, question):
    global IMAGE_MODEL
    global VQA_MODEL
    global TOKENIZER
    global graph

    # Open image
    img = get_image_from_url(img_url)

    im = prepare_image(img, img_shape=(150, 150))

    # Generate feature maps from image model
    image_features = np.zeros((1, 4096))
    with graph.as_default():
        image_features[0, :] = IMAGE_MODEL.predict(im)[0]

    # Generate embeddings or input question
    question_token = TOKENIZER.texts_to_sequences([question])
    question_features = pad_tokens(question_token, 26)

    # Predict from VQA model
    with graph.as_default():
        pred = VQA_MODEL.predict([question_features, image_features])[0]

    # Filter top 5 answers
    top_preds = pred.argsort()[-5:][::-1]

    return [(ANSWERS_IDX.at[idx, 'answer'].capitalize(), round(pred[idx] * 100.0, 2)) for idx in top_preds]


def prepare_image(image, img_shape=(224, 224)):
    im = resize_image(image, img_shape)
    im = np.transpose(im, (1, 2, 0))
    #print(im.shape)

    # this axis dimension is required because VGG was trained on a dimension
    im = np.expand_dims(im, axis=0)

    return im


def pad_tokens(df_serie, max_len=300):
    return pad_sequences(df_serie, padding='post', maxlen=max_len)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_URL'] = '../uploads/template.jpg'


@app.route("/")
def template_test():
    return render_template('index.html', imagesource='/static/home.jpg', question='', anwer='')


@app.route('/predict_vqa', methods=['POST'])
def predict_vqa():
    #if request.method == 'POST':
    file = request.files['photo']

        #if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # print(file_path)
    app.config['IMAGE_URL'] = file_path

    text = request.form['textbox']
    question = text.capitalize()
    # print(question)

    preds = predict(app.config['IMAGE_URL'], question)
    #preds = predict(file_path, question)
    # print(preds)

    return render_template('index.html', imagesource=app.config['IMAGE_URL'], question=question, answer=preds, answer_title="Here are my answers")


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['photo']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            #print(file_path)
            app.config['IMAGE_URL'] = file_path

            return render_template('index.html', imagesource=app.config['IMAGE_URL'], question='', anwer='')


@app.route('/ask_question', methods=['POST'])
def ask_question():
    text = request.form['textbox']
    question = text.capitalize()
    #print(question)

    preds = predict(app.config['IMAGE_URL'], question)
    #print(preds)

    return render_template('index.html', imagesource=app.config['IMAGE_URL'], question=question, answer=preds)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOAD_FOLDER']
})


def init():
    global IMAGE_MODEL
    global VQA_MODEL
    global TOKENIZER
    global graph

    IMAGE_MODEL = keras.models.load_model(f"{MODEL_FOLDER}{IMG_MODEL_FILE}")
    VQA_MODEL = keras.models.load_model(f"{MODEL_FOLDER}{MODEL_FILE}")
    with open(f"{MODEL_FOLDER}{TOKENIZER_FILE}", 'rb') as handle:
        TOKENIZER = pickle.load(handle)

    graph = tf.get_default_graph()


if __name__ == "__main__":
    app.debug = False
    init()
    app.run(host='0.0.0.0', port=3000)
