# 👁️ Diabetic Retinopathy Classification using Deep Learning

This project aims to **detect and classify the severity of Diabetic Retinopathy (DR)** from retinal fundus images using deep learning models. A Flask web application was also developed to deploy the best-performing models for real-time predictions.

---

## 📌 Project Overview

**Diabetic Retinopathy (DR)** is a complication of diabetes that affects the eyes and can lead to blindness if not detected early. This project uses **Convolutional Neural Networks (CNNs)** to automatically classify retina images into five stages:

- **0 - No_DR**: No signs of diabetic retinopathy
- **1 - Mild_DR**
- **2 - Moderate_DR**
- **3 - Severe_DR**
- **4 - Proliferative_DR**

We trained multiple CNN architectures on a Kaggle dataset and achieved high performance in both accuracy and F1-score.

---

## 📊 Achievements

- 🧠 Trained and evaluated models: `Xception`, `ResNet101`, `ResNet152`, `ResNet50`, and `VGG16`
- 🏆 Achieved **94% classification accuracy** on the best-performing models
- 📈 Boosted **F1-Score to 0.92** using:
  - **K-Fold Cross-Validation**
  - **Data Augmentation**
  - **Regularization Techniques**
- 🌐 Developed a **Flask Web App** to allow users to upload retinal images and get predictions using `VGG16` and `ResNet101`

---

## 📁 Dataset

- **Source**: [Kaggle - Diabetic Retinopathy Detection](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
- **Format**: Fundus images with 5 classification labels:
  - `0`: No_DR
  - `1`: Mild_DR
  - `2`: Moderate_DR
  - `3`: Severe_DR
  - `4`: Proliferative_DR

### 🔍 Preprocessing Techniques

- Resizing images to standard input size
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Top-hat** and **Black-hat** morphological operations to enhance fine blood vessel visibility

---

## 🧪 Model Architectures

| Model       | Accuracy | F1-Score | Notes                        |
|-------------|----------|----------|------------------------------|
| ResNet101   | 94%      | 0.92     | Deployed in the final app    |
| VGG16       | 92%      | 0.89     | Lightweight + great accuracy |
| ResNet152   | 93%      | 0.90     | Heavy, slower in prediction  |
| Xception    | 90%      | 0.86     | Good depthwise performance   |
| ResNet50    | 91%      | 0.87     | Faster, slightly less accurate |

---

## 🧰 Techniques Used

- ✅ **Image Preprocessing**:
  - CLAHE
  - Histogram Equalization
  - Top-hat & Black-hat Enhancements
- ✅ **Model Optimization**:
  - Batch Normalization
  - Dropout Regularization
  - Learning Rate Scheduling
  - K-Fold Cross Validation
- ✅ **Data Augmentation**:
  - Rotation
  - Flipping
  - Scaling
  - Brightness & Contrast Adjustments

---

## 💻 Flask Web Application

An intuitive **Flask-based web app** was created for real-time prediction:

- 📷 Upload a fundus image via web UI
- 📊 App displays predicted DR severity level
- 🧠 Backend: Pretrained `VGG16` and `ResNet101` models
- 🎨 Frontend: Bootstrap-powered interface


---

## 🏗️ Project Structure

diabetic-retinopathy-classifier/
│
├── app/
│ ├── static/
│ ├── templates/
│ ├── app.py
│ └── model_utils.py
│
├── models/
│ ├── vgg16_model.h5
│ ├── resnet101_model.h5
│
├── notebooks/
│ ├── EDA.ipynb
│ ├── model_training.ipynb
│ └── performance_analysis.ipynb
│
├── data/
│ └── (image samples)
│
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy
Edit

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/diabetic-retinopathy-classifier.git
cd diabetic-retinopathy-classifier
```
2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
```
2️⃣  Install Dependencies
```bash
pip install -r requirements.txt
```
📦 Requirements
Python 3.8+

TensorFlow / Keras
OpenCV
Flask
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn

```bash

pip install -r requirements.txt
```
🧑‍💻 Author
Harshil Handoo

🧠 ML Developer | Data Scientist

💼 Built DR detector, deployed in Flask app


📌 Future Improvements
📊 Add Grad-CAM visualization to interpret model decisions

🧠 Incorporate ensemble methods for robust predictions

🐳 Dockerize the application for easy deployment

📱 Optimize models for mobile deployment

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

If you found this project helpful, feel free to ⭐ star the repo and fork for enhancements!
