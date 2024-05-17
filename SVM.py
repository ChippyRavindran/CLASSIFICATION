import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import pickle

# Step 1: Data loading and preprocessing
data_dir = "C:/Users/User/Desktop/chippy/Feature_extraction_machine_learning/DATASET/dataset_1/dataset_full"
categories = ["Glacier", "Mountains", "Sea", "Streets", "Forest", "Building"]
image_size = (100, 100)  # Resize images to a standard size

X, y = [], []
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

for category_id, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    images = [img_name for img_name in os.listdir(folder_path) 
              if os.path.splitext(img_name)[1].lower() in valid_image_extensions]
    images = images[:500] if category != "Forest" else images[:2745]
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
        img = cv2.resize(img, image_size)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flattened_img = gray_img.flatten()
        X.append(flattened_img)
        y.append(category_id)

X = np.array(X)
y = np.array(y)

# Step 2: Feature extraction
histograms = [np.histogram(image, bins=256, range=(0, 256))[0] for image in X]
features = np.array(histograms)

# Step 3: Dimensionality reduction
pca = PCA(n_components=100)
features_reduced = pca.fit_transform(features)

# Step 4: Classification algorithm
X_train, X_test, y_train, y_test = train_test_split(features_reduced, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
svm_classifier.fit(X_train_scaled, y_train)

# Step 5: Evaluation
y_pred = svm_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, target_names=categories))

# Step 6: Save the trained model, PCA, and scaler
model_dir = 'C:/Users/User/Desktop/chippy/models/'
os.makedirs(model_dir, exist_ok=True)

model_filename = os.path.join(model_dir, 'svm_model.pkl')
pca_filename = os.path.join(model_dir, 'pca.pkl')
scaler_filename = os.path.join(model_dir, 'scaler.pkl')

with open(model_filename, 'wb') as model_file:
    pickle.dump(svm_classifier, model_file)
with open(pca_filename, 'wb') as pca_file:
    pickle.dump(pca, pca_file)
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model, PCA, and scaler saved.")




import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained SVM model, PCA, and scaler
model_dir = 'C:/Users/User/Desktop/chippy/models/'
model_filename = os.path.join(model_dir, 'svm_model.pkl')
pca_filename = os.path.join(model_dir, 'pca.pkl')
scaler_filename = os.path.join(model_dir, 'scaler.pkl')

with open(model_filename, 'rb') as model_file:
    svm_classifier = pickle.load(model_file)
with open(pca_filename, 'rb') as pca_file:
    pca = pickle.load(pca_file)
with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load and preprocess the input image
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
def classify_image(image_path):
    features = extract_features(image_path)
    if features is None:
        return None
    
    scaled_features = scale_features(features)
    
    if scaled_features.shape[1] != 100:
        print("Error: Invalid number of dimensions in scaled features.")
        return None
    
    class_label = svm_classifier.predict(scaled_features)
    return categories[class_label[0]]

# Path to the input image
image_path = 'C:/Users/User/Desktop/chippy/TEST_IMAGES/1b.jpg'

# Classify the input image
predicted_class = classify_image(image_path)
if predicted_class is not None:
    print("Predicted class:", predicted_class)

