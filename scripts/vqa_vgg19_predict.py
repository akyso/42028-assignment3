import tensorflow as tf
import tensorflow.keras as keras

print(tf.__version__)
print(tf.keras.__version__)

import pandas as pd
import numpy as np
import pickle
import h5py
import json

valid_df = pd.read_hdf('dataframe_val.h5', 'val')

# Load Question Features

with open('question_padded_token_val.pkl', 'rb') as handle:
    question_padded_token_val = pickle.load(handle)

# Load Image Features

img_feat_val_hf = h5py.File('vgg19_img_features_val.h5', 'r')
img_feat_val = img_feat_val_hf.get('val')
img_feat_val = np.array(img_feat_val)
img_id_val = pd.DataFrame(img_feat_val_hf.get('image_id'))
print(img_id_val.shape)
print(img_feat_val.shape)

from keras.models import Sequential, Model

model_name = f'models/vqa_vgg19/vqa_vgg19.h5'

model = keras.models.load_model(model_name)

def get_img_feat(df, img_id, img_feat):
    img_mapping = img_id.reset_index()
    img_mapping.columns = ['index', 'image_id']
    img_list = df.merge(img_mapping, how='left', on='image_id')['index'].tolist()
    return img_feat[img_list]

test_img_feature = get_img_feat(valid_df, img_id_val, img_feat_val)
test_X  = [question_padded_token_val, test_img_feature]

step_size = valid_df.shape[0]
#print(test_X.shape)

preds = model.predict(test_X, verbose=1)
#print(preds.shape)

answer_mapping = pd.read_hdf('answer_mapping.h5', 'answers')

preds_idx = preds.argmax(axis=-1).tolist()
preds_name = answer_mapping.iloc[preds_idx,]
preds_name = preds_name.reset_index(drop=True)

valid_df['preds'] = preds_name['answer']

answer_list = []
for pred, question_id, question in zip(valid_df['preds'], valid_df['question_id'], valid_df['question']):
	#answer_list.append({"answer": pred, "question_id": question_id})
	print(f"question: {question}\nanswer: {pred}")

answer_json = json.dumps(answer_list)

with open(f'vqa_vgg19_val_preds.json', 'w') as outfile:
   json.dump(answer_json, outfile)
