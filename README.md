# 🚦 GTSRB Traffic Sign Recognition (Real Dataset)

This project implements a **real traffic sign recognition system** using a  
**Convolutional Neural Network (CNN)** trained on the official  
**GTSRB – German Traffic Sign Recognition Benchmark** dataset.  
The system supports **model training** and **real-time traffic sign detection**
via a webcam.

✅ Real dataset (GTSRB)  
✅ Real CNN training  
✅ Real-time webcam inference  
❌ No synthetic or fake data  

---

## 📂 Project Structure

```text
.
├── main.py
├── README.md
└── gtsrb/                    # NOT included in repository
    └── Training/
        ├── 00000/
        ├── 00001/
        ├── ...
        └── 00042/
⚠️ The dataset and trained model files are intentionally excluded from GitHub
due to file size and dataset licensing restrictions.

📦 Dataset (Required)
This project uses the official GTSRB dataset published by the
German Traffic Sign Recognition Benchmark.

🔗 Download link:
https://benchmark.ini.rub.de/gtsrb_dataset.html

After downloading, extract the dataset and place it in the project root
with the following structure:

text
Kodu kopyala
gtsrb/Training/00000/*.ppm
Each folder (00000 – 00042) represents one traffic sign class

Images must be in .ppm format (original dataset format)

🧠 Model Architecture
Type: Convolutional Neural Network (CNN)

Input size: 48 × 48 RGB

Number of classes: 43

Optimizer: Adam

Loss function: Sparse Categorical Crossentropy

Regularization:

Batch Normalization

Dropout layers

📁 Trained model file:

text
Kodu kopyala
gtsrb_real_model.h5
🎓 Model Training
If the trained model file does not exist, the system will
automatically start training when executed.

bash
Kodu kopyala
python main.py
Training features:

Data augmentation (rotation, zoom, brightness)

Validation split

Early stopping

Best model checkpoint saving

⏱ Training time depends on hardware and dataset size
(typically 20–30 minutes on a standard GPU system).

🎥 Real-Time Traffic Sign Detection
If gtsrb_real_model.h5 already exists, the program directly starts
the webcam-based detection mode.

The system:

Detects red traffic sign candidate regions

Crops detected regions

Resizes them to the model input size

Classifies them using the trained CNN

Displays class name and confidence score in real time

🎮 Controls
Q / ESC → Exit

SPACE → Save screenshot

+ / - → Increase / decrease confidence threshold

⚠️ Notes & Limitations
Dataset and trained .h5 model file are not included

Intended for educational and research purposes

Detection accuracy depends on:

Lighting conditions

Camera quality

Distance and angle of the traffic sign

Real-world deployment requires further optimization

📜 License
GTSRB dataset license belongs to its original authors

Source code is free to use for educational and non-commercial purposes

✅ Summary
This repository provides:

A complete real dataset training pipeline

A real-time traffic sign detection system

Clear instructions to reproduce results locally

Simply download the dataset, place it in the correct folder,
and run the project to get started.
