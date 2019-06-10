import pandas as pd
import numpy as np
import pickle
import h5py

training_df= pd.read_hdf('dataframe_train.h5', 'train')
valid_df = pd.read_hdf('dataframe_val.h5', 'val')

# Load Question Features

with open('question_padded_token_train.pkl', 'rb') as handle:
    question_padded_token_train = pickle.load(handle)

with open('question_padded_token_val.pkl', 'rb') as handle:
    question_padded_token_val = pickle.load(handle)

# Load Image Features

img_feat_train_hf = h5py.File('vgg19_img_features_train.h5', 'r')
img_feat_train = img_feat_train_hf.get('train')
img_feat_train = np.array(img_feat_train)
img_id_train = pd.DataFrame(img_feat_train_hf.get('image_id'))
print(img_id_train.shape)
print(img_feat_train.shape)

img_feat_val_hf = h5py.File('vgg19_img_features_val.h5', 'r')
img_feat_val = img_feat_val_hf.get('val')
img_feat_val = np.array(img_feat_val)
img_id_val = pd.DataFrame(img_feat_val_hf.get('image_id'))
print(img_id_val.shape)
print(img_feat_val.shape)

# Train VQA Model

from keras.models import Sequential, Model
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import Input, LSTM, Multiply, Dense, Embedding, Flatten
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

def get_cnn():
    model_image_in = Input(shape=(4096,))
    X1 = Dense(1024)(model_image_in)
    X1 = Activation('relu')(X1)
    model_image_out = Dropout(0.5)(X1)
    
    model_image = Model(model_image_in, model_image_out)
    
    return model_image, model_image_in, model_image_out
  
def get_lstm():
    model_language_in = Input(shape=(26,))
    X2 = Embedding(12603, 300, input_length=26)(model_language_in)
    X2 = LSTM(512, return_sequences=True, input_shape=(26, 300))(X2)
    X2 = LSTM(512, return_sequences=True)(X2)
    X2 = LSTM(512, return_sequences=False)(X2)
    X2 = Dense(1024)(X2)
    X2 = Activation('tanh')(X2)
    model_language_out = Dropout(0.5)(X2)
                              
    model_language = Model(model_language_in, model_language_out)
    
    return model_language, model_language_in, model_language_out

def create_DeeperLSTM():
  
    # Image model   
    model_image, model_image_in, model_image_out = get_cnn()
    
    # Language Model
    model_language, model_language_in, model_language_out = get_lstm()    
    
    # Merge models
    merged_in = concatenate([model_language_out, model_image_out])

    X = Dense(2048)(merged_in)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(1024)(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
        
    X = Dense(1000)(X)
    merged_out = Activation('softmax')(X)

    model = Model([model_language_in, model_image_in], merged_out)
    
    return model

model = create_DeeperLSTM()

def get_callbacks(file_path, chkpnt=True, estop=True, red_lr=True, csv_log=True, cp_name="-{epoch:04d}-{val_loss:.2f}.ckpt"):
  
    callbacks = []

    if chkpnt:
      callbacks.append(ModelCheckpoint(file_path + cp_name, monitor="val_loss", mode="min",\
                        save_weights_only=True, save_best_only=True, verbose=1))
    if estop:
      callbacks.append(EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=5,\
                        restore_best_weights=True, verbose=1))
    if red_lr:
      callbacks.append(ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3,\
                        min_delta=0.00001, verbose=True))
    if csv_log:
      callbacks.append(CSVLogger(file_path + '-training.log'))

    return callbacks

callbacks = get_callbacks("models/vqa_vgg19")

lr_start = 1e-4
rmsprop = RMSprop(lr=lr_start)
adam = Adam(lr=lr_start)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

def get_img_feat(df, img_id, img_feat):
    img_mapping = img_id.reset_index()
    img_mapping.columns = ['index', 'image_id']
    img_list = df.merge(img_mapping, how='left', on='image_id')['index'].tolist()
    return img_feat[img_list]

train_img_feature = get_img_feat(training_df, img_id_train, img_feat_train)
test_img_feature = get_img_feat(valid_df, img_id_val, img_feat_val)

train_X = [question_padded_token_train, train_img_feature]
test_X  = [question_padded_token_val, test_img_feature]

train_Y = np_utils.to_categorical(training_df[u'answer_id'], 1000)
test_Y = np_utils.to_categorical(valid_df[u'answer_id'], 1000)

print(model.summary())

history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), \
                callbacks=callbacks, batch_size=32, nb_epoch=1)

model.save(f'models/vqa_vgg19/vqa_vgg19.h5')
