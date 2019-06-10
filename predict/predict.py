import tensorflow as tf
import tensorflow.keras as keras

from predict.get_features import get_image_features, get_question_features, get_image_model
from predict.preprocess_image import get_image_from_url
from settings import ANSWERS_IDX, MODEL_FOLDER, MODEL_FILE, IMG_MODEL_FILE


def predict(img_url, question, image_model=None, vqa_model=None):
    # Open image
    img = get_image_from_url(img_url)

    # Load image model
    if not image_model:
        image_model = load_image_model()

    # Generate feature maps from image model
    image_features = get_image_features(img, image_model, img_shape=(150, 150))

    # Generate embeddings or input question
    question_features = get_question_features(question)

    # Load VQA Model
    if not vqa_model:
        vqa_model = load_vqa_model()

    # Predict from VQA model
    pred = vqa_model.predict([question_features[:, :, 0], image_features])[0]

    """
    # Filter top 5 answers
    top_pred = pred.argsort()[-5:][::-1]

    # Map answers from indexes
    return [(ANSWERS_IDX[_].title(), round(pred[_] * 100.0, 2)) for _ in top_pred]
    """

    pred_idx = pred.argmax(axis=-1).tolist()
    pred_name = ANSWERS_IDX.iloc[pred_idx,]
    pred_name = pred_name.reset_index(drop=True)

    return [pred_name.title(), round(pred[_] * 100.0, 2)]


def load_image_model():
    if IMG_MODEL_FILE != "":
        return keras.models.load_model(f"{MODEL_FOLDER}{IMG_MODEL_FILE}")
    else:
        return get_image_model()


def load_vqa_model():
    return keras.models.load_model(f"{MODEL_FOLDER}{MODEL_FILE}")