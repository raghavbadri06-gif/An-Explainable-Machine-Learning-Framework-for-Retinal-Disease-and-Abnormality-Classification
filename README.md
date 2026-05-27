# Early Detection of Retinal Diseases and Abnormalities Using Machine Learning

> **Note:** This work is currently **under consideration** and has been submitted to ***Sensing and Imaging***. Please do not distribute or cite without author permission until publication is confirmed.

## Overview

This study aims to identify retinal diseases and abnormalities at an early stage, thereby preventing serious visual loss or blindness using a Machine Learning (ML)-based method by analyzing fundus images. The proposed framework integrates preprocessing, feature extraction (structural, non-structural, and textural), and classification using three ML models: **Support Vector Machine (SVM)**, **Linear Discriminant Analysis (LDA)**, and **Bagged Trees (BT)**.

Among these, SVM showed the most consistent performance after addressing class imbalance using the **Synthetic Minority Oversampling Technique (SMOTE)**, achieving **96.80% accuracy** in identifying diseased eyes when trained and tested on the proposed feature set.

Additionally, a deep learning comparison is provided using **ResNet50**, and **SHAP (SHapley Additive exPlanations)** is used to identify the most influential features for clinical interpretability.
## Dataset Availability

The data used in this study were acquired from an open-source repository known as Kaggle, such as the Ocular Disease Intelligent Recognition (ODIR) and Joint Shantou International Eye Centre (JSIEC) dataset, whose respective links are given below:
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k?resource=download; 
https://www.kaggle.com/datasets/linchundan/fundusimage1000/data 

---


## Key Features

- **Preprocessing Pipeline:**
  - RGB to grayscale conversion
  - Gaussian filter
  - Median filter
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)

- **Feature Extraction:**
  - Structural features
  - Non-structural features
  - Textural features

- **Class Imbalance Handling:**
  - SMOTE (Synthetic Minority Oversampling Technique)

- **ML Classifiers Evaluated:**
  - Support Vector Machine (SVM)
  - Linear Discriminant Analysis (LDA)
  - Bagged Trees (BT)

- **Deep Learning Comparison:**
  - ResNet50

- **Model Interpretability:**
  - SHAP (SHapley Additive exPlanations)

---

## Results

| Model | Accuracy (with SMOTE) |
|-------|----------------------|
| SVM   | **96.80%**            |
| LDA   | Comparable but lower |
| BT    | Comparable but lower |

> SVM demonstrated the most consistent and reliable performance for early diagnosis in ophthalmological applications.

---

## Implementation Environment

The experiments were conducted using a dual-platform approach to leverage the strengths of both MATLAB and Python.

### Hardware Specifications used in this study

| Component               | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Workstation**        | Dell G15 Workstation                                                    |
| **Processor**          | 13th Gen Intel Core Ultra 9                                             |
| **Graphics Card**      | NVIDIA GeForce RTX 4080 Laptop GPU (12 GB GDDR6)                        |
| **RAM**                | 16 GB DDR5 (4800 MHz)                                                   |
| **Storage**            | 1 TB PCIe NVMe SSD                                                      |
| **Operating System**   | Windows 11 Pro (64-bit)                                                 |

### Software & Frameworks

#### MATLAB Environment (For ML Experiments)

- **Version:** MATLAB R2023a
- **Purpose:**
  - Preprocessing (filtering, grayscale conversion, CLAHE)
  - Feature extraction (structural, non-structural, textural)
  - Classification experiments via **MATLAB's Classification Learner App**
- **Advantage:** Superior image processing toolboxes and efficient feature engineering capabilities

#### Python Environment (For Deep Learning Experiments)

- **Version:** Python 3.10
- **Terminal/SSH Client:** MobaXterm (with CUDA-enabled graphics support)
- **Deep Learning Frameworks:** PyTorch / TensorFlow (CUDA-accelerated)
- **Purpose:**
  - ResNet50 implementation and evaluation
- **Advantage:** Access to state-of-the-art DL frameworks with powerful GPU acceleration and the code can be completed within a short span of time.

### Data Transfer Between Environments

- **Intermediate format:** CSV files
- Enables smooth, standardized transfer of extracted features from MATLAB (ML pipeline) to Python (DL pipeline)

---

## Workflow Summary

1. **Preprocessing (MATLAB R2023a)**  
   - Grayscale conversion → Gaussian → Median → CLAHE

2. **Feature Extraction (MATLAB R2023a)**  
   - Structural + Non-structural + Textural features

3. **Class Imbalance Handling**  
   - SMOTE

4. **ML Classification (MATLAB R2023a)**  
   - SVM, LDA, Bagged Trees (via Classification Learner App)

5. **Deep Learning Comparison (Python 3.10 + MobaXterm)**  
   - ResNet50 with CUDA acceleration

6. **Model Interpretability (Python 3.10)**  
   - SHAP analysis

---

## How to Reproduce

1. Place fundus images in the `data/` directory.
2. Run MATLAB scripts for preprocessing and feature extraction.
3. Export features as CSV files. (structural, non-structural and textural (12 features in total))
4. For ML classification, balance the feature set using the SMOTE code, where the dataset is split into train, test and validation
5. Then open MATLAB's classification learner app and then train the models using 5 fold cross validation and then import the test set, and get the results using these features: MeanIntensity, contrast, correlation, OpticCupArea, StdIntensity, skewness, RimThickness, kurtosis, and energy, which gives outperforming results, with the ML classifiers. 
6. . Additionally, It should be noted that 10% of the data was reserved as a validation set but ultimately not used, as the MATLAB Classification Learner App performs internal cross-validation for model selection. This represents an inefficiency in data utilisation; future work will employ a simple train-test split without a separate validation set .
7. Run Python scripts for ResNet50 training and testing analysis.
8. Compare results with the provided benchmarks.

---

## Publication Status

This work has been submitted to:

- **Journal:** ***Sensing and Imaging***
- **Status:** Under Consideration

---

## Citation

If you wish to reference this work (pending acceptance), please use:

```bibtex
@unpublished{submitted_sensing_imaging_retinal_ml,
  title = {Early Detection of Retinal Diseases and Abnormalities Using Machine Learning},
  author = {[Author Names]},
  note = {Submitted to Sensing and Imaging -- Under Consideration},
  year = {2026}
}
