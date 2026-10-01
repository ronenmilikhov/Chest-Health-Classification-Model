# Chest Health Classification

## Authors
* **Ronen Milikhov**
* **Shachar Wilk**
* Institution: Ruppin Academic Center
* Course: Medical Images Processing & Deep Learning

## Project Overview
This project develops a convolutional neural network (CNN) for classifying chest X-ray images into four categories.

The model classifies images into:
1. **COVID-19**
2. **Viral Pneumonia**
3. **Lung Opacity**
4. **Normal**

## Dataset
The notebook downloads the **COVID-19 Radiography Database** from Kaggle with `kagglehub`:
`tawsifurrahman/covid19-radiography-database`.

Images are loaded from the `COVID-19_Radiography_Dataset` directory using the four class folders `COVID`, `Normal`, `Viral Pneumonia`, and `Lung_Opacity`. The data is split into **80% training** and **20% validation** with `validation_split=0.2` and `seed=123`. There is no independent test set in the notebook.

## Architecture & Methodology
The notebook uses ImageNet-pretrained **ResNet50** without its original classification head. The ResNet50 backbone is fine-tuned because `base_model.trainable = True`.

**Key Technical Implementations:**
* **Input processing:** Images are resized to 224x224 and rescaled from [0, 255] to [0, 1]. No image augmentation is implemented.
* **Classification head:** `GlobalAveragePooling2D`, `Dense(128, activation='relu')`, `Dropout(0.2)`, and a four-unit output layer.
* **Class imbalance handling:** Class weights are calculated from the image counts in each class's `images` directory and passed to `model.fit`. Viral Pneumonia is the smallest counted class.
* **Data pipeline:** Training and validation datasets use `batch_size=32` and `tf.data.AUTOTUNE` prefetching.
* **Training:** Adam optimizer with learning rate `1e-3`, sparse categorical cross-entropy with `from_logits=True`, and 10 epochs.

## Results
The reported results are from the notebook's validation split, not from an independent test set. The final recorded metrics are:

* **Training accuracy:** 0.8254
* **Validation accuracy:** 0.8019
* **Validation loss:** 0.5484

**Classification report:**
| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| COVID | 0.65 | 0.69 | 0.67 |
| Normal | 0.85 | 0.69 | 0.76 |
| Viral Pneumonia | 0.82 | 0.91 | 0.86 |
| Lung Opacity | 0.96 | 0.82 | 0.89 |

Overall validation accuracy is approximately **80%**, with macro-average F1 of **0.79** and weighted-average F1 of **0.80**.

**ROC-AUC Scores:**
* COVID: **0.85**
* Normal: **0.73**
* Viral Pneumonia: **0.91**
* Lung Opacity: **0.99**

## How to Run
1. Open [Medical_Images_Processing_Final_Project.ipynb](Medical_Images_Processing_Final_Project.ipynb) in **Google Colab** or a compatible Jupyter environment.
2. Allocate a **T4 GPU** (Runtime -> Change runtime type).
3. Run all cells. The script uses `kagglehub` to automatically download the dataset into the runtime environment.
4. The trained model will be saved as `medical_chest_classifier.keras` in the root directory upon completion.
