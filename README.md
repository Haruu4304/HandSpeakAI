# HandSpeakAI 🤟 — American Sign Language Recognition using CNN & OpenCV

HandSpeakAI is a Machine Learning–based system that recognizes **American Sign Language (ASL)** hand gestures in real time and maps them to their corresponding alphabet characters. The project aims to assist communication for the hearing and speech impaired by translating hand signs into readable text using **Convolutional Neural Networks (CNN)** and **OpenCV**.

---

## 🔗 Project Link

GitHub Repository: [https://github.com/Haruu4304/HandSpeakAI.git](https://github.com/Haruu4304/HandSpeakAI.git)

---

## 📌 Features

- ✋ Recognizes American Sign Language hand gestures using a trained CNN model
- 🎯 Custom dataset collection and preprocessing pipeline for improved accuracy
- 🔤 Maps detected hand signs to their corresponding ASL characters
- 🎥 Real-time gesture detection and prediction using OpenCV
- 📈 Optimized model architecture for better prediction accuracy and usability

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – for real-time video capture and image processing
- **CNN (Convolutional Neural Network)** – for gesture classification
- **TensorFlow / Keras** (or equivalent deep learning framework used in training)
- **NumPy / Pandas** – for data handling and preprocessing

---

## 📂 Project Workflow

1. **Data Collection** – Capturing hand gesture images for each ASL character using OpenCV.
2. **Preprocessing** – Resizing, normalizing, and augmenting images to improve model robustness.
3. **Model Training** – Training a CNN on the custom dataset to classify hand signs.
4. **Real-Time Prediction** – Using OpenCV to capture live video feed and predict the corresponding ASL character in real time.
5. **Output Mapping** – Displaying the recognized ASL character as text output.

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Haruu4304/HandSpeakAI.git
cd HandSpeakAI

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

> Note: Update the file/script names above to match the actual entry point in your repository.

---

## 🚀 Usage

1. Run the script to start the webcam feed.
2. Position your hand within the detection frame.
3. Perform an ASL gesture — the model will predict and display the corresponding character in real time.

---

## 📊 Results

- Improved prediction accuracy through better preprocessing and dataset augmentation.
- Achieved smoother and more reliable real-time gesture recognition for assistive communication use cases.

---

## 🔮 Future Improvements

- Extend recognition from individual characters to full words and phrases.
- Improve model robustness across different lighting conditions and backgrounds.
- Deploy as a web or mobile application for wider accessibility.
- Add support for two-handed and dynamic (motion-based) ASL gestures.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository, raise issues, or submit pull requests to improve the project.

---

## 📄 License

This project is open-source. Feel free to use and modify it for educational and assistive purposes.

---

## 👤 Author

**Harsha**
GitHub: [@Haruu4304](https://github.com/Haruu4304)
