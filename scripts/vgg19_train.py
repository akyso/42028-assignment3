import tensorflow as tf
import tensorflow.keras as keras

print(tf.__version__)
print(tf.keras.__version__)

from keras.applications.vgg19 import VGG19
from keras.models import Model,Sequential
from keras.layers import Input, LSTM, Multiply, Dense, Embedding, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, Adam

import json
import pandas as pd
import numpy as np
import glob
import os
#import spacy
import h5py
import pickle

def list_images_from_dir(path):
  #img_list = glob.glob(path + '/*.jpg')
  img_list = os.listdir(path)
  img_list = [img.replace(path, '') for img in img_list]
  print(len(img_list))
  return img_list

"""
def remove_missing_images(df, col_name, img_list):
  return df[df[col_name].isin(img_list)]

#with open('data/coco/annotations/instances_train2014.json') as json_file:  
with open('data/coco/annotations/annotations/instances_train2014.json') as json_file:  
    train_data = json.load(json_file)

images_df = pd.DataFrame(train_data['images'])
annotations_df = pd.DataFrame(train_data['annotations'])
categories_df = pd.DataFrame(train_data['categories'])
images_df = images_df[['file_name','height','id','width']]
annotations_df = annotations_df[['category_id','image_id']]
categories_df = categories_df[['id', 'name']]

train_set = annotations_df.merge(images_df, how='left', left_on='image_id', right_on='id').drop('id', axis=1)
train_set = train_set.merge(categories_df, how='left', left_on='category_id', right_on='id').drop('id', axis=1)
"""
train_dir = "data/coco/images/train2014/"
train_img_list_file = 'data/coco/images/train2014_img_list.pkl'

train_images = list_images_from_dir(train_dir)
with open(train_img_list_file, 'wb') as fp:
    pickle.dump(train_images, fp)

train_set = pd.DataFrame()
train_set['file_name'] = pd.Series(train_images)
train_set['name'] = "hot dog"

#with open (train_img_list_file, 'rb') as fp:
#    train_images = pickle.load(fp)
#len(train_images)
"""
train_set = remove_missing_images(train_set, 'file_name', train_images)

train_set.drop_duplicates(subset=["file_name"], keep = False, inplace = True) 
print(train_set.shape)
print(train_set.head())
print(len(train_images))
"""
# Updated to do image augmentation
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=train_set, 
    directory=train_dir, 
    x_col="file_name", 
    y_col="name", 
    class_mode="categorical", 
    target_size=(150, 150), 
    batch_size=1)

model_name = f'models/vgg19/vgg19_feature_extractor.h5'

model_feat_extract = keras.models.load_model(model_name)

step_size_train=train_generator.n
print(f"number of step_size_train: {step_size_train}")
train_preds = model_feat_extract.predict_generator(train_generator, verbose=1, steps=step_size_train)

with h5py.File('vgg19_img_features_train.h5','w') as hf:
  hf.create_dataset('train', data=train_preds)
  hf.create_dataset('file_name', data=train_set['file_name'])

