# 🧠 Crack Detection using Machine Learning

This project detects cracks in surface images using a **Random Forest Classifier**. It classifies images into **positive (cracked)** and **negative (non-cracked)** classes. The model is trained on grayscale image features and deployed via a Streamlit web application.

---

## 🚀 Features

- Image upload via Streamlit UI
- Automatic crack detection (Positive or Negative)
- Uses preprocessed image features for fast predictions
- Trained with thousands of real-world images from civil infrastructure

---

## 🗂️ Project Structure

ai_crack/
│
├── app.py # Streamlit web application
├── model/
│ └── rf_model.py # Random Forest training script
│ └── rf_model.joblib # Trained model (auto-generated)
├── utils/
│ └── preprocess.py # Feature extraction and preprocessing
├── dataset/
│ ├── positive/ # Images with cracks
│ └── negative/ # Images without cracks
└── README.md


---

## 🧪 Dataset

We used the **SDNET2018** dataset from Missouri University of Science and Technology. It includes over 29,000 images of cracked and non-cracked surfaces.

📂 Folders:
- `positive/`: Contains cracked images
- `negative/`: Contains non-cracked images

🔗 **Download here**:  
[SDNET2018 Dataset – Mendeley](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

---

## ⚙️ How to Run

1. 🔧 Install requirements:
   ```bash
   pip install -r requirements.txt
#🧠 Train the model (if not already trained):
python -m model.rf_model

#🌐 Launch the Streamlit app:
streamlit run app.py
