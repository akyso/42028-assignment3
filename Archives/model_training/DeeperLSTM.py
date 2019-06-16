from keras.models import Sequential, Model
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import Input, LSTM, Multiply, Dense, Embedding, Flatten
from keras.layers import concatenate


def get_cnn(args):
    model_image_in = Input(shape=(args['img_vec_dim'],))
    X1 = Dense(args['num_hidden_units_mlp'])(model_image_in)
    X1 = Activation(args['activation_1'])(X1)
    model_image_out = Dropout(args['dropout'])(X1)

    model_image = Model(model_image_in, model_image_out)

    return model_image, model_image_in, model_image_out


def get_lstm(args):
    model_language_in = Input(shape=(args['max_ques_length'],))
    X2 = Embedding(args['vocabulary_size'], args['word_emb_dim'], input_length=args['max_ques_length'])(
        model_language_in)
    X2 = LSTM(args['num_hidden_units_lstm'], return_sequences=True,
              input_shape=(args['max_ques_length'], args['word_emb_dim']))(X2)
    X2 = LSTM(args['num_hidden_units_lstm'], return_sequences=True)(X2)
    X2 = LSTM(args['num_hidden_units_lstm'], return_sequences=False)(X2)
    X2 = Dense(args['num_hidden_units_mlp'])(X2)
    X2 = Activation(args['activation_1'])(X2)
    model_language_out = Dropout(args['dropout'])(X2)

    model_language = Model(model_language_in, model_language_out)

    return model_language, model_language_in, model_language_out


def create_DeeperLSTM(args):
    # Image model
    model_image, model_image_in, model_image_out = get_cnn(args)

    # Language Model
    model_language, model_language_in, model_language_out = get_lstm(args)

    # Merge models
    merged_in = concatenate([model_language_out, model_image_out])

    for i in range(args['num_hidden_units_mlp']):
        X = Dense(args['num_hidden_units_mlp'])(merged_in)
        X = Activation(args['activation_1'])(X)
        X = Dropout(args['dropout'])(X)

    X = Dense(args['nb_classes'])(X)
    merged_out = Activation(args['class_activation'])(X)

    model = Model([model_language_in, model_image_in], merged_out)

    return model
