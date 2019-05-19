from __future__ import print_function
import numpy as np
import h5py
import json
import pandas as pd


def get_data(args, split='train'):
    data = {}

    img_norm = args['img_normalize']

    # Load json file
    dataset = load_input_json(args)

    # load image feature
    img_feature = load_img_feature(args, split)

    # load h5 file
    data = load_qa_feature(args, split)

    if img_norm:
        print('(get_data) Normalizing image feature')
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem, (1, args['img_vec_dim'])))

    if split == "train":
        val_answers = None
    else:
        val_answers = get_val_answers(args, data, dataset)

    return dataset, img_feature[data['img_list']], data, val_answers


def load_input_json(args):
    dataset = {}

    with open(args['input_json']) as data_file:
        print('(get_data) Loading input json file...')
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    return dataset


def load_img_feature(args, split):
    img_feature = None

    img_split = f'images_{split}'

    with h5py.File(args['input_img_h5'], 'r') as hf:
        print('(get_data) Loading image feature...')
        # -----0~82459------
        tem = hf.get(img_split)
        img_feature = np.array(tem)

    return img_feature


def load_qa_feature(args, split):
    data = {}

    ques_split = f"ques_{split}"
    ques_length_split = f"ques_length_{split}"
    img_pos_split = f"img_pos_{split}"
    question_id_split = f"question_id_{split}"

    if split == "train":
        answers_split = f"answers"
    else:
        answers_split = f"MC_ans_test"

    with h5py.File(args['input_ques_h5'], 'r') as hf:
        print('(get_data) Loading h5 file...')

        # total number of model_training data is 215375
        # question is (26, )
        tem = hf.get(ques_split)
        data['question'] = np.array(tem)
        print(f"(get_data - {split}) Nb questions: {len(data[u'question'])}")

        # max length is 23
        tem = hf.get(ques_length_split)
        data['length_q'] = np.array(tem)

        # total 82460 img
        # -----1~82460-----
        tem = hf.get(img_pos_split)
        # convert into 0~82459
        data['img_list'] = np.array(tem) - 1
        print(f"(get_data - {split}) Nb images: {len(data[u'img_list'])}")

        # quiestion id
        tem = hf.get(question_id_split)
        data['ques_id'] = np.array(tem)

        # answer is 1~1000
        tem = hf.get(answers_split)
        data['answers'] = np.array(tem) - 1
        print(f"(get_data - {split}) Nb answers: {len(data[u'answers'])}")

    return data


def get_val_answers(args, data, dataset):
    def most_common(lst):
        return max(set(lst), key=lst.count)

    # Added by Adi, make sure the ans_file is provided
    nb_data_test = len(data[u'question'])
    val_all_answers_dict = json.load(open(args['ans_file']))

    val_answers = np.zeros(nb_data_test, dtype=np.int32)
    ans_to_ix = {v: k for k, v in dataset[u'ix_to_ans'].items()}
    count_of_not_found = 0

    for i in range(nb_data_test):
        qid = data[u'ques_id'][i]
        try:
            val_ans_ix = int(ans_to_ix[most_common(val_all_answers_dict[str(qid)])]) - 1
        except KeyError:
            count_of_not_found += 1
            val_ans_ix = 480
        val_answers[i] = val_ans_ix
    print("(get_data - test) Beware: " + str(count_of_not_found) + " number of val answers not found")

    return val_answers