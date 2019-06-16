
def get_arguments():
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
        'ans_file': 'data/val_all_answers_dict.json',
        'input_json': 'data/data_prepro.json',
        'input_img_h5': 'data/data_img.h5',
        'input_ques_h5': 'data/data_prepro.h5',

        # pre-model_training
        'weights': '',
        'model_h5': '',

    }

    return args

