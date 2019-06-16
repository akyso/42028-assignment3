# Folders Structure

```
├── README.md
├── .gitignore                  <- Ignore files and folders
├── app.py                      <- Flask app (interface for VisualQA)
├── settings.py                 <- Contains global variables
├── data
│   ├── preprocessed            <- Pre-processed images (extracted feature maps from vgg16)
│   ├── vqa_v1                  <- Annotations (questions & answers) and image list from Visual QA
│   └── coco                    <- The original images, immutable data dump.
│
├── models                      <- Folder containing models h5 files used for prediction 
│
├── notebooks                   <- Jupyter notebooks
│
├── scripts                     <- Folder containing functions to predict VQA result
│   ├── vgg19_model.py          <- Functions to extract faetures map and embeddings
│   ├── vgg19_train.py          <- Predict from feature maps and embeddings
│   ├── vgg19_val.py            <- Function to pre-process images
│   ├── vqa_vgg19_bow.py
│   ├── vqa_vgg19_model.py
│   └── vqa_vgg19_predict.py    <- Function to pre-process images
│
├── templates                   <- HTML files used by Flask app
│   └── index.html              <- HTML file containing Jinja variables       
│
└─── uploads                    <- Folder containing images uploaded from Flask app
```