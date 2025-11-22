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


## Quantization Aware Training (QAT)
The notebook includes an optional Quantization Aware Training (QAT) section at the end that demonstrates preparing the trained MobileNetV2 for int8 quantization using PyTorch’s FX graph mode APIs.

- What it does
  - Creates a CPU copy of the trained FP32 model and evaluates a baseline (top‑1 accuracy + a small latency sample per batch on CPU).
  - Prepares the model for QAT with get_default_qat_qconfig('fbgemm') and FX prepare_qat_fx.
  - Optionally runs a brief QAT fine‑tuning loop on a small subset to calibrate/fine‑tune fake‑quant blocks.
  - Converts the model to an int8 quantized version with convert_fx and evaluates accuracy and latency again on CPU.
  - Saves artifacts for comparison and optional deployment.

- How to run it
  1) Open FruitClassification.ipynb and scroll to the section titled “Quantization Aware Training (QAT) with Pytorch”.
  2) Set the flag RUN_QAT = True in that section to enable the brief QAT fine‑tuning before conversion (leave it False to only perform baseline and conversion steps that don’t require fine‑tuning).
  3) Run the cells in that section. The code will automatically choose the best device for each stage (see below).

- Devices and performance
  - QAT fine‑tuning stage: runs on CUDA if available, otherwise CPU. This stage uses fake‑quant modules that can run on GPU for speed.
  - Conversion and inference stage: runs on CPU. PyTorch’s quantized int8 kernels (fbgemm/qnnpack) are CPU‑only.

- Outputs produced
  - Console printout with baseline FP32 accuracy/latency and post‑quantization int8 accuracy/latency (measured on one sample batch repeated a few times).
  - Saved models in the QATmodels folder (created automatically):
    - QATmodels/model_fp32_state_dict.pth (FP32 weights)
    - QATmodels/model_int8_qat_fx_state_dict.pth (INT8 weights produced by FX QAT)
    - QATmodels/model_int8_qat_fx_scripted.pt (optional TorchScript export if scripting succeeds)

- Compatibility and notes
  - The notebook uses a compatibility shim to import FX QAT APIs from torch.ao.quantization.quantize_fx (newer PyTorch) or torch.ao.quantization.fx (older layout). If neither is available, the section is skipped with a clear message.
  - To reduce noise, known deprecation warnings from the legacy torch.ao.quantization API and observer configuration are suppressed within the QAT block only. The rest of the notebook isn’t affected.
  - PyTorch upstream notes that torch.ao.quantization (FX) is being deprecated; future migrations should consider torchao’s PT2E path (prepare_pt2e/convert_pt2e). The current example intentionally sticks to widely available FX APIs for simplicity.

If you see a message that FX QAT APIs are unavailable, install a compatible PyTorch/torchvision build (see requirements.txt comments for CPU/CUDA guidance) and re‑run the section.


## Use‑case ideas from the dataset description
The original dataset page lists several potential applications such as:
- dietary management for fruit allergies 
- wellness apps
- supermarket checkout assistance
- agriculture quality control
- and educational uses. 

For best results, use images that clearly contain fruit objects relevant to the trained classes.

