import tensorflow as tf
import tensorflow.keras as keras

from predict.get_features import get_image_features, get_question_features, get_image_model
from predict.preprocess_image import get_image_from_url
from settings import ANSWERS_IDX, MODEL_FOLDER, MODEL_FILE, IMG_MODEL_FILE


def predict(img_url, question):
    # Open image
    img = get_image_from_url(img_url)

    # Load image model
    if IMG_MODEL_FILE != "":
        image_model = keras.models.load_model(f"{MODEL_FOLDER}{IMG_MODEL_FILE}")
    else:
        image_model = get_image_model()

    # Generate feature maps from image model
    image_features = get_image_features(img, image_model)

    # Generate embeddings or input question
    question_features = get_question_features(question)

    # Load VQA Model
    model = keras.models.load_model(f"{MODEL_FOLDER}{MODEL_FILE}")

    # Predict from VQA model
    pred = model.predict([question_features[:, :, 0], image_features])[0]

    # Filter top 5 answers
    top_pred = pred.argsort()[-5:][::-1]

    # Map answers from indexes
    #return [(ANSWERS_IDX[_].title(), round(pred[_] * 100.0, 2)) for _ in top_pred]
    return [(ANSWERS_IDX[_].title(), round(pred[_] * 100.0, 2)) for _ in top_pred]