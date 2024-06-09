
from flask import Flask, request, jsonify, render_template
from skimage.feature import local_binary_pattern
import cv2 as cv
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model, PCA model, and label encoder
model = joblib.load('knn_set3_pca.pkl')
pca = joblib.load('pca_model_optimal.pkl')
label_encoder = joblib.load('label_encoder_pca.pkl')


def preprocess_image(image):
    resized_image = cv.resize(image, (150, 150))
    gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image

def extract_histogram(image):
    # Extract color histogram features
    hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

def extract_sobel_edges(image):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.hypot(sobelx, sobely)
    sobel_combined = cv.normalize(sobel_combined, sobel_combined).flatten()
    return sobel_combined

def extract_lbp(image):
    # Extract Local Binary Patterns (LBP) features
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

def extract_entropy(image):
    entropy = -np.sum(image * np.log2(image + 1e-10))
    return np.array([entropy])

def extract_hog(image):
    # Extract Histogram of Oriented Gradients (HOG) features
    hog = cv.HOGDescriptor()
    h = hog.compute(image)
    return h.flatten() # type: ignore

def extract_corners(image):
    corners = cv.cornerHarris(image, 2, 3, 0.04)
    return corners.flatten()

def extract_features_set1(image, image_type=None):
    processed_image = preprocess_image(image)
    color_image = cv.cvtColor(cv.resize(image, (150, 150)), cv.COLOR_BGR2RGB)
    combined_features = []
    edges = extract_sobel_edges(processed_image)
    texture = extract_lbp(processed_image)
    color_hist = extract_histogram(color_image)
    combined_features = np.hstack([edges, texture, color_hist])
    return combined_features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    # Save the file to the static directory
    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    
    # Read image file
    image = cv.imread(file_path)
    
    # Extract features for prediction
    features = extract_features_set1(image)

    # Determine the max_length used during training
    max_length = 23022  # Replace this with the actual max length used during training

    # Pad the features to the max_length
    features = np.pad(features, (0, max_length - len(features)), 'constant')
    
    # Apply PCA for dimensionality reduction
    features = pca.transform([features])  # Ensure features is in the correct shape for PCA

    # Predict
    prediction = model.predict(features)
    
    # Decode prediction
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    return render_template('index.html', prediction=predicted_label, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)