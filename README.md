# Malaria Parasite Detection — Classical Computer Vision Pipeline

Blood cell image classification using hand-crafted features and scikit-learn classifiers, built as a university Computer Vision Lab assignment.

---

## Overview

Malaria is diagnosed by examining Giemsa-stained blood smear slides under a microscope and manually counting infected red blood cells. This project builds a fully classical image classification pipeline that labels a single segmented cell image as **Parasitized** or **Uninfected**, the core step in automating that process.

The pipeline uses no deep learning. All feature extraction is done with OpenCV and scikit-image, and all classifiers are from scikit-learn, making every step interpretable and reproducible without a GPU.

---

## Dataset

**Cell Images for Detecting Malaria**  
Originally released by the U.S. National Institutes of Health (NIH / National Library of Medicine).  
Kaggle mirror: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria  
Licence: Public domain (NIH)

| Property | Value |
|---|---|
| Classes | Parasitized, Uninfected |
| Total images | 27,558 |
| Images per class | 13,779 (balanced) |
| Image type | Giemsa-stained thin blood smear, single-cell crops |
| Download size | ~340 MB |

A stratified random subset of 3,000 images per class (6,000 total) is used by default to keep grid searches manageable. Set `SUBSET_PER_CLASS = None` in the configuration cell to use the full dataset.

The dataset is not included in this repository. The notebook downloads it automatically with `kagglehub` on first run.

**Citation**  
Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., Jaeger, S., Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ* 6:e4568. https://doi.org/10.7717/peerj.4568

---

## Pipeline

```
raw images
    └─ duplicate removal (exact MD5 match)
    └─ stratified 70 / 15 / 15 split (train / val / test)
         └─ preprocessing
               resize 64 × 64  →  median filter  →  CLAHE (L channel)  →  grayscale  →  normalise [0, 1]
         └─ feature extraction
               HOG  +  LBP (radius 1 & 2)  +  GLCM  +  HSV histogram   →  1,880-dimensional vector
         └─ Pipeline(scaler, selector, classifier)   ×   GridSearchCV (5-fold, macro F1)
               Selectors : None | Chi-square (SelectKBest) | L1-SVM (SelectFromModel)
               Classifiers: KNN | SVM-RBF | Random Forest
         └─ evaluation on test set (accuracy, precision, recall, F1, AUC)
```

---

## Notebook structure

| Section | Contents |
|---|---|
| 0 | Dataset justification |
| 1 | Dataset description — class distribution chart, sample grid, image-size stats, duplicate check |
| 2 | Subset, 70/15/15 stratified split, five preprocessing steps with before/after images and CLAHE histogram |
| 3 | Feature extraction (HOG, LBP, GLCM, HSV) and feature-count table before selection |
| 4 | Nine Pipeline + GridSearchCV combinations (3 selectors × 3 classifiers), feature-count table after selection |
| 5 | Evaluation — combined results table, confusion matrices, ROC curve, error analysis with misclassified images |
| 6 | Discussion |

---

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
opencv-python-headless
scikit-image
scikit-learn
Pillow
joblib
kagglehub
```

Install everything at once:

```bash
pip install numpy pandas matplotlib opencv-python-headless scikit-image scikit-learn pillow joblib kagglehub
```

The notebook was developed on **Python 3.13** with scikit-learn 1.9.1 and scikit-image 0.26.0. It runs on Google Colab and Kaggle Notebooks without changes.

---

## Usage

**On Kaggle (easiest)** — add the dataset to your notebook session directly from the Data tab, then set:

```python
DATA_DIR = "/kaggle/input/cell-images-for-detecting-malaria"
```

**On Google Colab** — mount a Kaggle token and let `kagglehub` download automatically, or upload the dataset to Google Drive and point `DATA_DIR` at it.

**Locally** — clone this repo, install the requirements, and run:

```bash
jupyter notebook RollNo_MalariaCells.ipynb
```

On the first run `kagglehub` will download the dataset (~340 MB) and cache it. Subsequent runs load extracted features from a compressed `.npz` cache file, skipping re-extraction.

Set `QUICK_RUN = True` in the configuration cell to use small hyperparameter grids for a fast end-to-end test.

---

## Known issues

- `cv2.CLAHE` cannot be pickled by joblib's loky (multiprocessing) backend. The fix is to call `cv2.createCLAHE()` inside `step_clahe` rather than at module level, and to use `Parallel(prefer="threads")` in `extract_many` instead of the default process backend.
- The Kaggle download includes a nested copy of the class folders. The duplicate-removal cell (Section 1) handles this automatically.
- Because the cells are pre-segmented and cleaned, test scores are optimistic relative to raw blood smear images.

---

## Repository structure

```
.
├── RollNo_MalariaCells.ipynb   ← main notebook with all outputs
├── RollNo_MalariaCells.pdf     ← PDF export for submission
├── results_all_combinations.csv← saved from Section 5
└── README.md
```

The dataset and cached feature files (`features_*.npz`, `pipeline_cache/`) are excluded via `.gitignore`.

---

## .gitignore

```
features_*.npz
pipeline_cache/
*.csv
__pycache__/
.ipynb_checkpoints/
```

---

## Licence

Code in this repository is released under the MIT licence. The dataset is in the public domain (NIH / NLM).
