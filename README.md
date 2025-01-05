# **EMOTION DETECTION USING MEDIAPIPE & RANDOM FOREST**



This project demonstrates facial expression detection (anger, disgust, fear, happiness, sadness, surprise) using:

MediaPipe Holistic to capture face landmarks
scikit-learn RandomForest for classification
It does not require TensorFlow/Keras, making it a relatively lightweight solution for prototyping or demonstrations.

Features
Data Collection

Captures 30-frame sequences of facial landmarks per emotion class.
Landmarks are saved as .npy files for each frame.
Model Training

Loads and flattens the landmark data (468 × 3 coords × 30 frames) into a single vector.
Uses a RandomForestClassifier to learn how to distinguish each emotion.
Real-Time Inference

Classifies live webcam input by maintaining a rolling buffer of 30 frames.
Displays predictions on the webcam feed.
Requirements
Python 3.7+
pip (or conda) environment
Libraries:
numpy
opencv-python
mediapipe
scikit-learn
matplotlib (optional, for plotting if desired)
Install them using:

bash
Copy code
pip install numpy opencv-python mediapipe scikit-learn matplotlib
Repository Structure
bash
Copy code
emotion-detection/
  ├── face_emotion_rf.py          # Main script for data collection, training, and inference
  ├── README.md                   # Project documentation (this file)
  └── MP_Face_Data_Sklearn/       # Generated folder after data collection, containing:
      ├── anger/
      ├── disgust/
      ├── fear/
      ├── happiness/
      ├── sadness/
      └── surprise/
          └── sequences of .npy files
Usage
Clone the Repository

Clone or download this project to your local machine.
bash
Copy code
git clone https://github.com/<your-username>/emotion-detection.git
cd emotion-detection
Install Dependencies

Use either a virtual environment or install directly:
bash
Copy code
pip install opencv-python mediapipe scikit-learn matplotlib
Collect Data (Optional)

By default, the script can record 5 sequences of 30 frames for each of the 6 emotions.
Open face_emotion_rf.py, look near the bottom in the if __name__ == "__main__": block:
python
Copy code
# collect_data()
Uncomment that line to collect new data. Then run:
bash
Copy code
python face_emotion_rf.py
A webcam window will open, prompting you to record frames for each emotion in actions.
Press q if you ever need to quit early.
Train the Model

After collecting data, re-comment the collect_data() line to avoid re-collecting.
Run:
bash
Copy code
python face_emotion_rf.py
The script will:
Load all .npy files from MP_Face_Data_Sklearn/.
Split them into training/testing sets.
Train a RandomForestClassifier.
Print the confusion matrix, classification report, and accuracy.
Real-Time Detection (Optional)

Uncomment real_time_detection(model) in the same if __name__ == "__main__": block:
python
Copy code
# real_time_detection(model)
Run again:
bash
Copy code
python face_emotion_rf.py
A webcam window will open and continuously classify the last 30 frames of face data as one of the six emotions.
Press q to exit.
Troubleshooting
Webcam permission issues on macOS

Check System Settings (or System Preferences on older versions) → Privacy & Security → Camera.
Ensure that the Terminal or your IDE is allowed to use the camera.
No data loaded:

Means you did not collect data or the folder structure is incomplete.
Ensure each sequence folder has 0.npy through 29.npy.
Low Accuracy:

Collect more sequences per class or increase your data quality.
Consider fine-tuning RandomForest hyperparameters or using more frames.
License
This project is released under the MIT License (or add a different license file if desired).

Contributing
Contributions, bug reports, and feature requests are welcome. Feel free to open an issue or create a pull request.

