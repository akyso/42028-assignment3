import tensorflow.keras as keras
import pickle

from Archives.predict.get_features import get_image_features, get_question_features, get_image_model
from Archives.predict.preprocess_image import get_image_from_url
from settings import ANSWERS_IDX, MODEL_FOLDER, MODEL_FILE, IMG_MODEL_FILE, TOKENIZER_FILE


def predict(img_url, question, image_model=None, vqa_model=None, tokenizer=None):

    # Open image
    img = get_image_from_url(img_url)

    # Load image model
    if not image_model:
        image_model = load_image_model()

    # Generate feature maps from image model
    image_features = get_image_features(img, image_model, img_shape=(150, 150))

    if not tokenizer:
        tokenizer = load_tokenizer()

    # Generate embeddings or input question
    question_features = get_question_features(question, tokenizer)

    # Load VQA Model
    if not vqa_model:
        vqa_model = load_vqa_model()

    # Predict from VQA model
    pred = vqa_model.predict([question_features, image_features])[0]

    # Filter top 5 answers
    top_preds = pred.argsort()[-5:][::-1]

    return [(ANSWERS_IDX.at[idx, 'answer'], round(pred[idx] * 100.0, 2)) for idx in top_preds]


def load_image_model():
    if IMG_MODEL_FILE != "":
        return keras.models.load_model(f"{MODEL_FOLDER}{IMG_MODEL_FILE}")
    else:
        return get_image_model()


def load_vqa_model():
    return keras.models.load_model(f"{MODEL_FOLDER}{MODEL_FILE}")


def load_tokenizer():
    with open(f"{MODEL_FOLDER}{TOKENIZER_FILE}", 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

