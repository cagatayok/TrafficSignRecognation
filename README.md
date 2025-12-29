# 🚦 GTSRB Traffic Sign Recognition (Real Dataset)

This project trains a **Convolutional Neural Network (CNN)** using the official  
**GTSRB – German Traffic Sign Recognition Benchmark** dataset and performs  
**real-time traffic sign detection** via webcam.

✅ Real dataset  
✅ Real CNN training  
✅ Real-time inference  
❌ No synthetic or fake data  

---

## 📂 Project Structure

.
├── main.py
├── README.md
└── gtsrb/              # (NOT included in repository)
    └── Training/
        ├── 00000/
        ├── 00001/
        ├── ...
        └── 00042/

⚠️ Dataset and trained model files are intentionally excluded from GitHub
due to size and licensing restrictions.

---

## 📦 Dataset (Required)

This project uses the **official GTSRB dataset**.

Download link:
https://benchmark.ini.rub.de/gtsrb_dataset.html

Expected folder structure:
gtsrb/Training/00000/*.ppm

Each folder (00000–00042) represents a traffic sign class.

---

## 🧠 Model Details

- Convolutional Neural Network (CNN)
- Input size: 48x48 RGB
- Number of classes: 43
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Regularization: Batch Normalization & Dropout

Trained model file:
gtsrb_real_model.h5

---

## 🎓 Training

If the trained model does not exist, training starts automatically.

python main.py

Training includes:
- Data augmentation (rotation, zoom, brightness)
- Early stopping
- Best model checkpointing

---

## 🎥 Real-Time Detection

If `gtsrb_real_model.h5` exists, webcam detection can be started directly.

Controls:
- Q / ESC → Exit
- SPACE → Screenshot
- + / - → Confidence threshold adjustment

The system detects red traffic sign regions, crops them,
and classifies them using the trained CNN.

---

## ⚠️ Notes

- Dataset and `.h5` model file are not included in this repository.
- Intended for educational and research purposes.
- Webcam performance depends on lighting conditions.

---

## 📜 License

Dataset license belongs to the GTSRB authors.
Source code is free to use for educational purposes.
