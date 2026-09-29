# Facial Expression Classification

A deep learning project for classifying human facial expressions into 7 different emotion classes using a Convolutional Neural Network (CNN).

## Overview

This project uses the **FER-2013 dataset** to train a CNN model that recognizes facial expressions from images.

The model classifies images into:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

The project also includes a simple **Streamlit web application** that allows users to upload an image and get a predicted facial expression.

## Dataset

The project uses the **FER-2013 dataset**.

The dataset contains grayscale facial images with a resolution of **48 × 48 pixels**.

After cleaning the dataset, the images were divided into training, validation, and test sets.

## Model

The model uses a custom CNN architecture consisting of:

* Convolutional layers
* Batch Normalization
* ReLU activation
* Max Pooling
* Dropout
* Global Average Pooling
* Dense layers
* Softmax output layer

Input shape:

```text
48 × 48 × 1
```

Output:

```text
7 emotion classes
```

## Training

The images were normalized from:

```text
0 - 255
```

to:

```text
0 - 1
```

The model was trained using the **Adam optimizer** with sparse categorical crossentropy as the loss function.

Early stopping and model checkpointing were also used to save the best model based on validation accuracy.

## Results

The final model achieved approximately:

| Metric              | Result |
| ------------------- | -----: |
| Validation Accuracy |  57.7% |
| Test Accuracy       |  57.1% |

The model performs differently across emotion classes. **Happy** and **Surprise** were among the classes with better recognition, while **Disgust** was difficult for the model to recognize because of its relatively small number of samples.

## Web Application

The project includes a Streamlit application where users can upload an image and receive a predicted facial expression.

The application performs:

1. Image upload
2. Face detection
3. Image preprocessing
4. Facial expression prediction
5. Display of the predicted emotion

### Run the application

Clone the repository:

```bash
git clone https://github.com/BertoPurba/KlasifikasiEkspresi.git
```

Enter the project directory:

```bash
cd KlasifikasiEkspresi
```

Create and activate the virtual environment:

```bash
python -m venv .venv
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

The application will then be available at:

```text
http://localhost:8501
```

## Project Structure

```text
KlasifikasiEkspresi/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── processed/
│   ├── train_split.csv
│   └── val_split.csv
│
├── models/
│   └── improved_cnn_best.keras
│
├── notebooks/
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

## Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn
* OpenCV
* Streamlit
* Matplotlib
* Seaborn

## Future Improvements

Some possible improvements for this project include:

* Increasing the number of training samples for minority classes
* Using data augmentation
* Improving face detection and cropping
* Experimenting with transfer learning
* Improving the model's performance on real-world images
* Deploying the Streamlit application online

## Author

**Berto Purba**

GitHub: [BertoPurba](https://github.com/BertoPurba)
