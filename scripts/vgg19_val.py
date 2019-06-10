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

#with open('data/coco/annotations/instances_val2014.json') as json_file:  
with open('data/coco/annotations/annotations/instances_val2014.json') as json_file:  
    data = json.load(json_file)

images_df = pd.DataFrame(data['images'])
annotations_df = pd.DataFrame(data['annotations'])
categories_df = pd.DataFrame(data['categories'])
images_df = images_df[['file_name','height','id','width']]
annotations_df = annotations_df[['category_id','image_id']]
categories_df = categories_df[['id', 'name']]

val_set = annotations_df.merge(images_df, how='left', left_on='image_id', right_on='id').drop('id', axis=1)
val_set = val_set.merge(categories_df, how='left', left_on='category_id', right_on='id').drop('id', axis=1)
"""
val_dir = "data/coco/images/val2014/"
val_img_list_file = 'data/coco/images/val2014_img_list.pkl'

val_images = list_images_from_dir(val_dir)
with open(val_img_list_file, 'wb') as fp:
    pickle.dump(val_images, fp)

val_set = pd.DataFrame()
val_set['file_name'] = pd.Series(val_images)
val_set['name'] = "hot dog"

val_set['image_id'] = val_set['file_name'].str.replace("COCO_val2014_", "")
val_set['image_id'] = val_set['image_id'].str.replace(".jpg", "")
val_set['image_id'] = pd.to_numeric(val_set['image_id'])

print(val_set.head())

#with open (val_img_list_file, 'rb') as fp:
#    val_images = pickle.load(fp)
#len(val_images)
"""
val_set = remove_missing_images(val_set, 'file_name', val_images)
val_set.drop_duplicates(subset=["file_name"], keep = False, inplace = True) 
print(val_set.shape)
"""
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator=test_datagen.flow_from_dataframe(
    dataframe=val_set, 
    directory=val_dir, 
    x_col="file_name", 
    y_col="name", 
    class_mode="categorical", 
    target_size=(150, 150), 
    batch_size=1)

model_name = f'models/vgg19/vgg19_feature_extractor.h5'

model_feat_extract = keras.models.load_model(model_name)

step_size_val=validation_generator.n
print(f"number of step_size_val: {step_size_val}")
val_preds = model_feat_extract.predict_generator(validation_generator, verbose=1, steps=step_size_val)

with h5py.File('vgg19_img_features_val.h5','w') as hf:
  hf.create_dataset('val', data=val_preds)
  hf.create_dataset('image_id', data=val_set['image_id'])


