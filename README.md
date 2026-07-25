# Chest Health Classification: COVID-19 & Pneumonia Detection 🫁

**Composers:** Ronen Milikhov & Shachar Wilk  
**Institution:** Ruppin Academic Center (Class of 2026)  
**Domain:** Medical Image Processing & Deep Learning  

## Project Overview
This project focuses on developing a deep learning tool to assist medical professionals in the rapid screening of chest X-rays. Due to the heavy diagnostic load on healthcare systems, especially during pandemics, our AI-based system performs initial triage to detect urgent respiratory conditions.

The Convolutional Neural Network (CNN) automatically classifies 2D digital chest X-rays into four diagnostic categories:
1. **COVID-19**
2. **Viral Pneumonia**
3. **Lung Opacity**
4. **Normal**

## Dataset
We utilized the open-source **COVID-19 Radiography Database** from Kaggle, containing over 21,000 tagged X-ray images. 
* **Training Set:** 80%
* **Validation Set:** 20%
* *Note: Clinical metadata was intentionally excluded to ensure the model bases its predictions purely on computer vision feature extraction.*

## Architecture & Methodology
We implemented a **Transfer Learning** approach using the **ResNet50** architecture (pre-trained on ImageNet). 

**Key Technical Implementations:**
* **Data Preprocessing:** Resized images to 224x224 and normalized pixels to a [0-1] scale to prevent exploding gradients.
* **Fine-Tuning:** Unfroze the base model layers (`trainable=True`) to allow filters to specialize in delicate medical textures.
* **Class Imbalance Handling:** Calculated and injected strict `class_weights` during training to penalize errors in minority classes (e.g., Lung Opacity).
* **Memory Optimization:** Replaced heavy `.cache()` methods with `tf.data.AUTOTUNE` and `.prefetch()` to prevent RAM crashes on Google Colab T4 GPUs.

## Results
The model achieved an overall accuracy of **80%** on the validation set.  
**ROC-AUC Scores:**
* Lung Opacity: **0.99**
* Viral Pneumonia: **0.91**
* COVID-19: **0.85**
* Normal: **0.73**

## How to Run
1. Open `medical_images_processing_final_project.ipynb` in **Google Colab**.
2. Allocate a **T4 GPU** (Runtime -> Change runtime type).
3. Run all cells. The script uses `kagglehub` to automatically download the dataset into the runtime environment.
4. The trained model will be saved as `medical_chest_classifier.keras` in the root directory upon completion.
