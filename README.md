
Inspired by:
https://github.com/iamaaditya/VQA_Keras
https://github.com/iamaaditya/VQA_Demo

# Folders Structure

```
├── README.md
├── .gitignore              <- Ignore files and folders
├── app.py                  <- Flask app (interface for VisualQA)
├── settings.py             <- Contains global variables
├── data
│   ├── preprocessed        <- Pre-processed images (extracted feature maps from vgg16)
│   └── raw                 <- The original images, immutable data dump.
│
├── model_training          <- Containing functions to train models (image model and VQA)
│
├── models                  <- Folder containing models h5 files used for prediction 
│
├── notebooks               <- Jupyter notebooks
│
├── predict                 <- Folder containing functions to predict VQA result
│   ├── get_features.py     <- Functions to extract faetures map and embeddings
│   ├── predict.py          <- Predict from feature maps and embeddings
│   └── preprocess_image    <- Function to pre-process images
│
├── templates               <- HTML files used by Flask app
│   └── index.html          <- HTML file containing Jinja variables       
│
└─── uploads                 <- Folder containing images uploaded from Flask app
```