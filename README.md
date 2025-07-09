# 🧠 Dog vs Cat Image Classifier

**INTERN NAME**: BHARAT BHANDARI  
**INTERN ID**: CT04DF123  
**MENTOR**: NEELA SANTOSH  
**COMPANY**: CODTECH IT SOLUTIONS  
**DOMAIN**: MACHINE LEARNING  
**DURATION**: 4 WEEKS

This internship project focuses on building a binary image classification model using Convolutional Neural Networks (CNNs) to distinguish between images of cats and dogs. The implementation is done using TensorFlow and Keras, and the dataset is sourced from Kaggle.

---

## 📂 Dataset Description

- **Source**: [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)  
- **Type**: JPEG images of cats and dogs  
- **Image Size**: Resized to 256x256 pixels  
- **Classes**:  
  - `0`: Cat  
  - `1`: Dog  

The dataset is preprocessed using Keras' `ImageDataGenerator` for normalization and augmentation.

---

## 🛠️ CNN Model Overview

The architecture of the CNN is built using the Keras Sequential API. Here's a brief summary:

- **Input Shape**: (256, 256, 3)
- **Layers**:
  1. `Conv2D(32)` → `ReLU` → `MaxPooling2D`
  2. `Conv2D(64)` → `ReLU` → `MaxPooling2D`
  3. `Conv2D(128)` → `ReLU` → `MaxPooling2D`
  4. `Flatten`
  5. `Dense(128)` → `ReLU` → `Dropout(0.2)`
  6. `Dense(64)` → `ReLU` → `Dropout(0.2)`
  7. `Dense(1)` → `Sigmoid` (for binary output)

---

## ⚙️ Environment Setup

To get the project up and running, follow these steps:

### 1. Install Dependencies
```bash
pip install tensorflow keras numpy matplotlib
```

### 2. Download Dataset
Use Kaggle CLI:
```bash
kaggle datasets download -d salader/dogs-vs-cats
unzip dogs-vs-cats.zip
```

### 3. Train the Model
Inside the Jupyter notebook:
```python
model.fit(train_data, validation_data=val_data, epochs=10)
```

---

## 📊 Training Insights

- **Loss Function**: Binary Crossentropy  
- **Optimizer**: Adam  
- **Validation Split**: 20%  
- **Epochs**: 10  
- **Batch Size**: 32  
- **Result**: Achieved approximately 80% validation accuracy  
- **Model Saved As**: `dog_cat_classifier.h5`

Data augmentation techniques such as horizontal flipping and zoom were applied to improve generalization.

---

## 🎯 What I Learned

Through this project, I gained hands-on experience in:

- Building CNN models from scratch  
- Preprocessing image datasets for ML tasks  
- Applying dropout and data augmentation to prevent overfitting  
- Evaluating model performance using accuracy metrics

---

## 🚀 Future Work

To enhance this model, potential next steps include:

- Incorporating Transfer Learning (e.g., ResNet50, MobileNetV2)  
- Tuning hyperparameters like learning rate, dropout, batch size  
- Adding callbacks such as EarlyStopping or ReduceLROnPlateau  
- Exploring advanced data augmentation techniques

---

📌 **Developed by**: *Bharat Bhandari*  
