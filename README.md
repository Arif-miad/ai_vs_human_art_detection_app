

---

# 📚 AI vs Human Art Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27-green)](https://streamlit.io/)

---

## 🖼 Project Overview

This project is an **AI-powered image classification system** that detects whether a piece of art is **AI-generated** or **human-created**.
It uses a **Convolutional Neural Network (CNN)** trained on a curated dataset of AI Art and Real Art.

Users can **upload an image** through the Streamlit app and get **instant prediction** with confidence score.

---

## 📂 Dataset

* **Source**: Private curated dataset
* **Structure**:

```
Art/
├── AiArtData/     # AI-generated art images
└── RealArt/       # Human-created art images
```

* Train / Validation / Test split is performed:

```
Art_split/
├── train/
│   ├── AiArtData/
│   └── RealArt/
├── val/
│   ├── AiArtData/
│   └── RealArt/
└── test/
    ├── AiArtData/
    └── RealArt/
```

* Image preprocessing: resized to 128x128, normalized, augmented for training

---

## ⚙️ Features

* **Binary Classification**: AI Art vs Human Art
* **CNN Model**: 3 Convolutional layers + MaxPooling + BatchNormalization + Dropout
* **Data Augmentation**: rotation, flip, zoom, shear
* **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
* **Saved Model**: `.h5` file for Streamlit deployment

---

## 🚀 Streamlit App

* Upload an image (`jpg`, `jpeg`, `png`)
* View uploaded image (resized to 50% height)
* Get **prediction**: AI Art / Human Art
* See **confidence score**

---

### Run Locally

1. Clone the repo:

```bash
git clone https://github.com/yourusername/AI-vs-Human-Art-Detection.git
cd AI-vs-Human-Art-Detection
```

2. Create virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run Streamlit app:

```bash
streamlit run app.py
```

5. Upload image and see predictions in browser.

---

## 📈 Model Training

* CNN trained using TensorFlow/Keras
* Binary classification using **sigmoid** activation
* Loss: `binary_crossentropy`
* Optimizer: `Adam`
* Epochs: 15 (can be increased for better accuracy)

---

## 📊 Evaluation

* **Test Accuracy**: e.g., 95%
* **Confusion Matrix**:

|              | Pred AI | Pred Human |
| ------------ | ------- | ---------- |
| Actual AI    | 95      | 5          |
| Actual Human | 4       | 96         |

* **Classification Report**: Precision, Recall, F1-score

---

## 🛠 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **PIL / Pillow**
* **Streamlit**
* **NumPy / Matplotlib / Seaborn**

---

## 🔗 Deployment

* Deployed via **[Streamlit Cloud](https://aiartvshumanart.streamlit.app/)** 
* Direct GitHub repository → Connect → Deploy

---

## ✨ Future Improvements

* Add **multi-class detection** for different AI models
* Improve **UI with Streamlit widgets**
* Compress model for faster loading
* Add **confidence heatmaps / Grad-CAM** visualization

---

## 📜 License

This project is for **educational and research purposes**.
Do not use for commercial purposes without permission.

---


