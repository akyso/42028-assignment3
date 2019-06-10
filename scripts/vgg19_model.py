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

train_dir = "data/coco/images/train2014/"
train_img_list_file = 'data/coco/images/train2014_img_list.pkl'

train_images = list_images_from_dir(train_dir)
with open(train_img_list_file, 'wb') as fp:
    pickle.dump(train_images, fp)

#with open (train_img_list_file, 'rb') as fp:
#    train_images = pickle.load(fp)
#len(train_images)

train_set = remove_missing_images(train_set, 'file_name', train_images)

print(train_set.shape)
print(train_set['name'].nunique())
print(train_set.head())

# Updated to do image augmentation
#train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator=train_datagen.flow_from_dataframe(
    dataframe=train_set, 
    directory=train_dir, 
    x_col="file_name", 
    y_col="name", 
    class_mode="categorical", 
    target_size=(150, 150), 
    batch_size=32)



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

val_dir = "data/coco/images/val2014/"
val_img_list_file = 'data/coco/images/val2014_img_list.pkl'

val_images = list_images_from_dir(val_dir)
with open(val_img_list_file, 'wb') as fp:
    pickle.dump(val_images, fp)

#with open (val_img_list_file, 'rb') as fp:
#    val_images = pickle.load(fp)
#len(val_images)

val_set = remove_missing_images(val_set, 'file_name', val_images)

print(val_set.shape)
print(val_set['name'].nunique())
print(val_set.head())

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator=test_datagen.flow_from_dataframe(
    dataframe=val_set, 
    directory=val_dir, 
    x_col="file_name", 
    y_col="name", 
    class_mode="categorical", 
    target_size=(150, 150), 
    batch_size=32)

#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Reconnect the layers
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu', name='aux_output')(x)

predictions = Dense(80, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:-3]:
    layer.trainable = False

from keras.optimizers import RMSprop, Adam

lr_start = 1e-4
rmsprop = RMSprop(lr=lr_start)
adam = Adam(lr=lr_start)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

step_size_train=train_generator.n//train_generator.batch_size
step_size_val=validation_generator.n//validation_generator.batch_size

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=step_size_train,  # 2000 images = batch_size * steps
      epochs=5,
      #callbacks=callbacks,
      workers=1, use_multiprocessing=False,
      #validation_data=validation_generator,
      #validation_steps=step_size_val,  # 1000 images = batch_size * steps
      verbose=1)

model_name = f'models/vgg19/vgg19_feature_extractor.h5'

model_feat_extract = Model(inputs=model.input, outputs=model.get_layer("aux_output").output)
model_feat_extract.save(model_name)

with open(f'models/vgg19/vgg19_feature_extractor_history.pkl', 'wb') as fp:
    pickle.dump(history, fp)

from keras.utils import plot_model
plot_model(model, to_file='models/vgg19/vgg19_feature_extractor_plot.png')
