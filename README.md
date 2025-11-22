# Computer Vision – Fruit Classification

This repository contains a compact computer vision project built in PyTorch. It fine‑tunes a pretrained MobileNetV2 model to classify fruit types using images from a Kaggle dataset placed under the archive directory. Although the original dataset includes object detection annotations (bounding boxes), this project frames the task as single‑label image classification by deriving one label per image.


## What this project is about
- Goal: Train and evaluate a lightweight image classifier that predicts the dominant fruit class present in an image.
- Approach: Convert object detection annotations into a single class per image (the most frequent object class on that image), then fine‑tune MobileNetV2 for classification.
- Outputs: During evaluation, the notebook saves key visualizations to the project root: learning_curves.png, confusion_matrix_counts.png, confusion_matrix_normalized.png, and val_samples.png.


## Dataset
- Source: Allergic-fruit – Computer Vision on Kaggle
  https://www.kaggle.com/datasets/imtkaggleteam/allergic-fruit-computer-vision/data
- Needed to be downloaded manually from the Kaggle website and unzipped into the archive directory.
- Location in this repo: archive/train, archive/valid (and optionally archive/test)
- Format: Each split directory contains images and an _annotations.csv file with bounding box labels: filename,width,height,class,xmin,ymin,xmax,ymax
- Labeling for this project: for each image, we select the mode (most frequent) class appearing among its bounding boxes. Ties are broken by first occurrence.


## How it works (Notebook overview)
The main workflow lives in FruitClassification.ipynb and includes:
1) Data loading and preprocessing
   - Custom Dataset (FruitsFromAnnotations) that reads _annotations.csv and collates image–label pairs.
   - Transforms with torchvision (resize, horizontal flip for train, normalization).
2) Model
   - Pretrained MobileNetV2 (torchvision.models) with the final classifier layer replaced to match the number of classes.
3) Training and evaluation
   - Standard train/validation loop with cross‑entropy loss and Adam optimizer.
   - Metrics: accuracy per epoch; at the end precision, recall, F1 (macro) using sklearn.metrics.
   - Confusion matrices (counts and row‑normalized) with seaborn heatmaps.
4) Visualizations and artifacts
   - Learning curves for loss and accuracy.
   - Confusion matrices and a small grid of validation samples with predicted labels.
   - Saved image files: learning_curves.png, confusion_matrix_counts.png, confusion_matrix_normalized.png, val_samples.png.


## How to run
1) Create and activate a virtual environment (recommended) and install dependencies:
   - ```python -m venv venv```
   - ```.\venv\Scripts\activate```
   - ```pip install -r requirements.txt```
2) Start Jupyter (Notebook or Lab) in this folder and open FruitClassification.ipynb.
   - ```jupyter notebook```
3) Ensure your data is in archive/train and archive/valid with their respective _annotations.csv files (already expected by the notebook).
4) Run the notebook cells. On completion, check the generated PNGs in the project root.


## Repository structure
- FruitClassification.ipynb – the main notebook with the full training/evaluation workflow
- archive/ – dataset folder with train/ and valid/ splits (and optional test/)
- requirements.txt – installable dependencies
- GeneratedPhotos - folder for PNGs saved by the notebook


## Requirements
It is recommended to use a virtual environment with Python 3.10 to isolate the project dependencies.

Installing the packages from requirements.txt with the prompt: ```pip install -r requirements.txt```
will create an environment that supports CPU compatible packages, but the requirements.txt contains
comments that help set up an environment for: SM 5.0 → 9.0 AND RTX 50-series SM 12.0 thanks to CUDA 13.0’s PTX forward compatibility.

### GPU usage
If a CUDA‑enabled GPU (e.g., some NVIDIA models) and a CUDA build of PyTorch are available, the notebook uses the GPU. 
It prints device diagnostics at startup.

| SM Version | Architecture               |           Supported            |
|-----------:|----------------------------|:------------------------------:|
| **5.0**    | Maxwell                    |             ✔ Yes              |
| **5.2**    | Maxwell                    |             ✔ Yes              |
| **6.0**    | Pascal                     |             ✔ Yes              |
| **6.1**    | Pascal                     |             ✔ Yes              |
| **7.0**    | Volta                      |             ✔ Yes              |
| **7.2**    | Jetson Xavier              |             ✔ Yes              |
| **7.5**    | Turing                     |             ✔ Yes              |
| **8.0**    | Ampere                     |             ✔ Yes              |
| **8.6**    | Ampere Laptop GPUs         |             ✔ Yes              |
| **8.9**    | Ada Lovelace (RTX 40)      |             ✔ Yes              |
| **9.0**    | Hopper (H100)              |             ✔ Yes              |
| **12.0**   | Ada-Next (RTX 50 Series)   | ✔ Yes (PTX forward-compatible) |


## Use‑case ideas from the dataset description
The original dataset page lists several potential applications such as:
- dietary management for fruit allergies 
- wellness apps
- supermarket checkout assistance
- agriculture quality control
- and educational uses. 

For best results, use images that clearly contain fruit objects relevant to the trained classes.

