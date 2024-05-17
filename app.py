from flask import Flask, render_template, request, flash, redirect
import os
import webbrowser
from threading import Timer
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for Flask sessions
app.config['UPLOAD_FOLDER'] = 'images'

# Load the trained SVM model, PCA, and scaler
model_dir = 'trained_model'  # Assuming the models directory is in the same directory as app.py
model_filename = os.path.join(model_dir, 'svm_model.pkl')
pca_filename = os.path.join(model_dir, 'pca.pkl')
scaler_filename = os.path.join(model_dir, 'scaler.pkl')

with open(model_filename, 'rb') as model_file:
    svm_classifier = pickle.load(model_file)
with open(pca_filename, 'rb') as pca_file:
    pca = pickle.load(pca_file)
with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the route for uploading and classifying images
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class = classify_image(file_path)
            if predicted_class is not None:
                return render_template('result.html', result=predicted_class)
    return render_template('upload.html')

# Preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load the image.")
        return None
    img = cv2.resize(img, (100, 100))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flattened_img = gray_img.flatten()
    return flattened_img

# Extract features from the preprocessed image
def extract_features(image_path):
    flattened_img = preprocess_image(image_path)
    if flattened_img is None:
        return None
    histogram = np.histogram(flattened_img, bins=256, range=(0, 256))[0]
    return histogram

# Scale the features using the same scaler used during training
def scale_features(features):
    features_reshaped = features.reshape(1, -1)
    features_reduced = pca.transform(features_reshaped)
    scaled_features = scaler.transform(features_reduced)
    return scaled_features

# Classify the image using the trained SVM model
categories = ["Glacier", "Mountains", "Sea", "Streets", "Forest", "Building"]
def classify_image(image_path):
    features = extract_features(image_path)
    if features is None:
        return None
    
    scaled_features = scale_features(features)
    
    if scaled_features.shape[1] != 100:
        print("Error: Invalid number of dimensions in scaled features.")
        return None
    
    class_label = svm_classifier.predict(scaled_features)
    category_name = categories[class_label[0]]  # Get the category name
    return category_name


# Define the main function to run the Flask app
if __name__ == '__main__':
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000/upload")).start()
    app.run(debug=True)
