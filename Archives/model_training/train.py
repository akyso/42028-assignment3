import sys, os

import json
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import backend as K
import importlib


# custom
# from utils import get_data, arguments

def train_model(model_name=None, pretrained=False, lr=None):
    ########### Load arguments ###########

    args = load_params()

    ########### Load datasets ###########

    dataset, train_X, train_Y, test_X, test_Y = load_datasets(args)

    ########### Initialise model filenames ###########

    model_path, weights_name, json_path = initialise_filenames(args)

    ########### Load Pre-Trained Model ###########

    model = load_pretrained_model(args, model, model_path, weights_name, test_X, test_Y, pretrained, lr)

    ########### Train Model ###########
    callbacks = get_callbacks(file_path, cp_name=weights_name)
    model, history = fit_model(model, args, model_path, weights_name, train_X, train_Y, test_X, test_Y, callbacks)

    ########### Save Model ###########
    # Print model summary
    model.summary()

    # Save model into h5
    save_model_h5(model, f'{model_path}_full.h5')
    model.save_weights(f'{model_path}_weights.h5')
    save_model_json(model, f'{model_path}_arch.json')

    return history.history


def load_datasets(args):
    dataset, train_img_feature, train_data, _ = get_data(args, split="train")  # get_train_data(args)
    dataset, test_img_feature, test_data, val_answers = get_data(args, split="test")  # get_test_data(args)

    train_X = [train_data[u'question'], train_img_feature]
    train_Y = np_utils.to_categorical(train_data[u'answers'], args['nb_classes'])
    test_X = [test_data[u'question'], test_img_feature]
    test_Y = np_utils.to_categorical(val_answers, args['nb_classes'])

    print(f"\nTrain data: {len(train_X[0])} & {len(train_X[1])} - {len(train_Y)}\n")
    print(f"\nTest data: {len(test_X[0])} & {len(test_X[1])} - {len(test_Y)}\n")

    return dataset, train_X, train_Y, test_X, test_Y


def load_pretrained_model(args, model_name, model_path, weights_name, test_X, test_Y, pretrained=None, lr=None):
    model = create_model(model_name, args)

    if pretrained:
        model = load_model_weights(model_path + weights_name, model)
        loss, acc = model.evaluate(test_X, test_Y)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

        # To set learning rate
        if lr:
            K.set_value(model.optimizer.lr, 0.001)
    else:
        model.compile(loss='categorical_crossentropy', \
                      optimizer=args['optimizer'], metrics=['accuracy'])

    return model


def fit_model(model, args, model_path, weights_name, train_X, train_Y, test_X, test_Y):
    # Define Keras callbacks
    callbacks = get_callbacks(model_path, cp_name=weights_name)

    # Train model
    history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), \
                        callbacks=callbacks, batch_size=args['batch_size'], nb_epoch=args['nb_epoch'], \
                        )
    return model, history


def get_callbacks(file_path, chkpnt=True, estop=True, red_lr=True, csv_log=True,
                  cp_name="-{epoch:04d}-{val_loss:.2f}.ckpt"):
    callbacks = []

    if chkpnt:
        callbacks.append(ModelCheckpoint(file_path + cp_name, monitor="val_loss", mode="min", \
                                         save_weights_only=True, save_best_only=True, verbose=1))
    if estop:
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=5, \
                                       restore_best_weights=True, verbose=1))
    if red_lr:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, \
                                           min_delta=0.00001, verbose=True))
    if csv_log:
        callbacks.append(CSVLogger(file_path + '-model_training.log'))

    return callbacks


def create_model(model_name, args):
    if model_name == "DeeperLSTM":
        return create_DeeperLSTM(args)
    if model_name == "simple_mlp":
        return create_simple_mlp(args)


def load_latest_checkpoint(weights_dir):
    checkpoint_dir = os.path.dirname(weights_dir)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model


def load_model_weights(weights_path, model):
    model.load_weights(weights_path)
    return model


def save_model_json(model, json_path):
    model_json = model.to_json()
    with open(json_path, 'w') as outfile:
        outfile.write(model_json)


def load_model_json(json_path):
    # Read JSON
    json_thread = open(json_path, 'r')
    model_json = json_thread.read()
    json_thread.close()

    return tf.keras.models.model_from_json(model_json)


def load_model_json_weights(json_path):
    model = load_model_json(json_path)


def save_model_h5(model, file_name):
    model.save(file_name)


def load_model_h5(file_name):
    return tf.keras.models.load_model(file_name)
