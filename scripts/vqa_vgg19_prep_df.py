import json
import pandas as pd
import numpy as np
import glob
import os
#import spacy
import h5py
import pickle

# Prepare VQA Dataframes

def list_images_from_dir(path):
  img_list = glob.glob(path + '/*.jpg')
  img_list = [img.replace(path, '') for img in img_list]
  print(len(img_list))
  return img_list

def remove_missing_images(df, col_name, img_list):
  return df[df[col_name].isin(img_list)]

data_folder = "data/vqa_v1/"

train_annot_file   = f"{data_folder}/mscoco_train2014_annotations.json"
val_annot_file     = f"{data_folder}/mscoco_val2014_annotations.json"

train_open_quest_file   = f"{data_folder}OpenEnded_mscoco_train2014_questions.json" 
train_mult_quest_file   = f"{data_folder}MultipleChoice_mscoco_train2014_questions.json" 
val_open_quest_file     = f"{data_folder}OpenEnded_mscoco_val2014_questions.json" 
val_mult_quest_file     = f"{data_folder}MultipleChoice_mscoco_val2014_questions.json"  

train_img_folder   = "train2014"
val_img_folder     = "val2014"

## Load JSON Files

def load_json_file(file_path):
  with open(file_path, "r") as read_file:
    json_dict = json.load(read_file)
  return json_dict

def json_2_df(file_path):  
  if 'annot' in file_path:
    key = 'annotations'
  elif 'quest' in file_path:
    key = 'questions'
  else:
    print("[ERROR] JSON file should be annotations or questions")
    return None
  json_file = load_json_file(file_path)
  json_dict = json_file.get(key)
  json_df = pd.DataFrame(json_dict)
  return json_df

def get_ques_annot_json(train_annot_file, val_annot_file, train_open_quest_file, val_open_quest_file):
  train_annot_df      = json_2_df(train_annot_file)
  val_annot_df        = json_2_df(val_annot_file)
  train_open_quest_df = json_2_df(train_open_quest_file)
  val_open_quest_df  = json_2_df(val_open_quest_file)
  
  return train_annot_df, val_annot_df, train_open_quest_df, val_open_quest_df

train_annot_df, val_annot_df, train_open_quest_df, val_open_quest_df = get_ques_annot_json(train_annot_file, val_annot_file, train_open_quest_file, val_open_quest_file)

## Rename Image file

def get_image_filename(img_id, dataSubType):
  return f"COCO_{dataSubType}_{str(img_id).zfill(12)}.jpg"

train_annot_df['image_name'] = train_annot_df['image_id'].apply(lambda img_id: get_image_filename(img_id, dataSubType=train_img_folder))

val_annot_df['image_name'] =   val_annot_df['image_id'].apply(lambda img_id: get_image_filename(img_id, dataSubType=val_img_folder))

## Remove Missing Images

train_img_list_file = f"{data_folder}train2014_img_list.pkl"
with open (train_img_list_file, 'rb') as fp:
    train_images = pickle.load(fp)
print(len(train_images))

train_annot_df = remove_missing_images(train_annot_df, 'image_name', train_images)
print(train_annot_df.shape)
print(train_annot_df['image_name'].nunique())

val_img_list_file = f"{data_folder}val2014_img_list.pkl"
with open (val_img_list_file, 'rb') as fp:
    val_images = pickle.load(fp)
print(len(val_images))

val_annot_df = remove_missing_images(val_annot_df, 'image_name', val_images)
print(val_annot_df.shape)

## Map Answers

def get_anwer_mappings(df, ans_col='multiple_choice_answer'):
    answer_mapping = df[ans_col].value_counts().reset_index()[0:1000]
    answer_mapping.columns = ['answer_mapping', 'answer_freq']
    answer_mapping = answer_mapping[['answer_mapping']].to_dict()
    answer_mapping = answer_mapping.get('answer_mapping')
    inv_answer_mapping = {v: [k] for k, v in answer_mapping.items()}
    #inv_answer_mapping[np.nan] = [-1]
    answer_mapping = {k: [v] for k, v in answer_mapping.items()}
    #answer_mapping[-1] = [np.nan]

    return answer_mapping, inv_answer_mapping

def map_answers(df, inv_answer_mapping, ans_col='multiple_choice_answer'):
    df['answer_id'] = df[ans_col].replace(inv_answer_mapping)

    # Replace all string answers that are not in the top 1000 list
    mask = df['answer_id'].str.contains(r'^[0-9]+$')
    mask = ~(mask.fillna(True))
    df.loc[mask, 'answer_id'] = -1

    df['answer_id'] = df['answer_id'].astype('int')

    # Replace all answers with value over 1000
    df.loc[df['answer_id'] >= (len(inv_answer_mapping)-1), 'answer_id'] = -1

    assert len(inv_answer_mapping) >= df['answer_id'].nunique()

    return df['answer_id']

def filter_answer(df, unknown=-1):
    return df[df['answer_id'] > unknown]

answer_mapping, inv_answer_mapping = get_anwer_mappings(train_annot_df, ans_col='multiple_choice_answer')

train_annot_df['answer_id'] = map_answers(train_annot_df, inv_answer_mapping, ans_col='multiple_choice_answer')
train_annot_df = filter_answer(train_annot_df, -1)
print(train_annot_df.head())

val_annot_df['answer_id'] = map_answers(val_annot_df, inv_answer_mapping, ans_col='multiple_choice_answer')
val_annot_df = filter_answer(val_annot_df, -1)
print(val_annot_df.head())

answer_mapping_df = pd.DataFrame.from_dict(answer_mapping, orient='index', columns=['answer'])
inv_answer_mapping_df = pd.DataFrame.from_dict(inv_answer_mapping, orient='index', columns=['answer_id'])

train_annot_df[['image_id', 'question_id', 'image_name', 'answer_id']].to_hdf('annotations_train.h5', key='train', mode='w')
val_annot_df[['image_id', 'question_id', 'image_name', 'answer_id']].to_hdf('annotations_val.h5', key='val', mode='w')

train_annot_df= pd.read_hdf('annotations_train.h5', 'train')
val_annot_df = pd.read_hdf('annotations_val.h5', 'val')

answer_mapping_df.to_hdf('answer_mapping.h5', key='answers', mode='w')

## Define questions, answer, image table

def get_image_path(img_id, path):
    return f"{path}{img_id}"

def get_data_table(annot_df, open_quest_df, path=None):
    df = annot_df.merge(open_quest_df, how='left', on=['question_id', 'image_id'])

    if path:
        df['file_name'] = df['image_name'].apply(lambda img_id: get_image_path(img_id, path=path))

    return df

training_df = get_data_table(train_annot_df, train_open_quest_df)
valid_df    = get_data_table(val_annot_df,   val_open_quest_df)

print(valid_df.shape)
print(valid_df.head())

training_df.to_hdf('dataframe_train.h5', key='train', mode='w')
valid_df.to_hdf('dataframe_val.h5', key='val', mode='w')
