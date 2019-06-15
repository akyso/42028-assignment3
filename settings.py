import json
import pandas as pd
import os
#import sys

#sys.path.insert(0,'/Users/anthonyso/Documents/42028-assignment3/')

REPO_FOLDER = '/Users/anthonyso/Documents/42028-assignment3/'
MODEL_FOLDER = f'{REPO_FOLDER}models/'
MODEL_FILE = "vqa_vgg19/vqa_vgg19.h5" #"vqa_vgg19/vqa_vgg19.h5"
IMG_MODEL_FILE = "vgg19/vgg19_feature_extractor.h5" #"vgg19/vgg19_feature_extractor.h5"
TOKENIZER_FILE = "tokenizer/tokenizer.pkl"

args = {

        # model
        'model': 'DeeperLSTM',
        'num_hidden_units_mlp': 1024,
        'num_hidden_units_lstm': 512,
        'num_hidden_layers_mlp': 3,
        'num_hidden_layers_lstm': 1,
        'dropout': 0.5,
        'activation_1': 'tanh',
        'activation_2': 'relu',

        # model_training
        'seed': 1337,
        'optimizer': 'rmsprop',
        'nb_epoch': 300,
        'nb_iter': 200000,
        'model_save_interval': 19,
        'batch_size': 64,

        # language features
        'word_vector': 'glove',
        'word_emb_dim': 300,
        'vocabulary_size': 12603,
        'max_ques_length': 100,
        'data_type': 'TRAIN',

        # image features
        'img_vec_dim': 4096,
        'img_features': 'resnet',
        'img_normalize': 1,

        # evaluations
        'nb_classes': 1000,
        'class_activation': 'softmax',
        'loss': 'categorical_crossentropy',
        'save_folder': '',

        # data
        'ans_file': f'{REPO_FOLDER}data/preprocessed/val_all_answers_dict.json',
        'input_json': f'{REPO_FOLDER}data/preprocessed/data_prepro.json',
        'input_img_h5': f'{REPO_FOLDER}data/preprocessed/data_img.h5',
        'input_ques_h5': f'{REPO_FOLDER}data/preprocessed/data_prepro.h5',

        # pre-model_training
        'weights': '',
        'model_h5': '',

    }

#metadata = json.load(open(args['input_json'], 'r'))

#ANSWERS_IDX = pd.DataFrame([metadata.get('ix_to_ans')]).T.reset_index()
#ANSWERS_IDX['index'] = ANSWERS_IDX['index'].astype('int') - 1
#ANSWERS_IDX = ANSWERS_IDX.sort_values('index').set_index('index').to_dict()[0]

ANSWERS_IDX = pd.read_hdf('data/vqa_v1/answer_mapping.h5', 'answers')