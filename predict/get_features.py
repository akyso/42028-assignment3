import numpy as np
import spacy
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from predict.preprocess_image import resize_image


def get_image_model():
    from keras.applications.vgg16 import VGG16

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(150, 150, 3))

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(4096, activation='relu')(x)

    # this is the model we will train
    image_model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    image_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #image_model.summary()

    return image_model


def get_image_features(image, image_model, img_shape=(224, 224)):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation

    im = resize_image(image, img_shape)
    im = np.transpose(im, (1, 2, 0))
    #print(im.shape)

    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)

    image_features[0, :] = image_model.predict(im)[0]

    return image_features


def get_glove_question_features(question):
    ''' For a given question, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    word_embeddings = spacy.load('en_vectors_web_lg') #en-core-web-sm
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 26, 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


def pad_tokens(df_serie, max_len=300):
    from keras.preprocessing.sequence import pad_sequences

    return pad_sequences(df_serie, padding='post', maxlen=max_len)


def get_question_features(question, tokenizer):
    question_token = tokenizer.texts_to_sequences([question])
    question_padded_token = pad_tokens(question_token, 26)

    return question_padded_token

