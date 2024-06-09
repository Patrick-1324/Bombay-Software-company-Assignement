import os
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from skimage.measure import shannon_entropy
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


def preprocess_image(image):
    resized_image = cv.resize(image, (150, 150))
    gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    equalized_image = cv.equalizeHist(gray_image)
    blurred_image = cv.GaussianBlur(equalized_image, (5, 5), 0)
    return blurred_image


def extract_histogram(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist = cv.normalize(hist, hist).flatten()
    return hist

def extract_sobel_edges(image):
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    sobel_combined = np.hypot(sobelx, sobely)
    sobel_combined = cv.normalize(sobel_combined, sobel_combined).flatten()
    return sobel_combined

def extract_lbp(image):
    lbp = local_binary_pattern(image, 24, 8, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_entropy(image):
    entropy = -np.sum(image * np.log2(image + 1e-10))
    return np.array([entropy])

def extract_hog(image):
    hog = cv.HOGDescriptor()
    h = hog.compute(image)
    return h.flatten() # type: ignore

def extract_corners(image):
    corners = cv.cornerHarris(image, 2, 3, 0.04)
    return corners.flatten() 

def extract_features_set3(image, image_type):
    processed_image = preprocess_image(image)
    color_image = cv.cvtColor(cv.resize(image, (150, 150)), cv.COLOR_BGR2RGB)
    combined_features = []
    if image_type == 'Buildings':
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([texture, color_hist])
    elif image_type == 'Forest':
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([texture, color_hist])
    elif image_type == 'Glacier':
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([texture, color_hist])
    elif image_type == 'Mountains':
        texture = extract_lbp(processed_image)
        gradients = extract_hog(processed_image)
        combined_features = np.hstack([texture, gradients])
    elif image_type == 'Sea':
        texture = extract_lbp(processed_image)
        gradients = extract_hog(processed_image)
        combined_features = np.hstack([texture, gradients])
    elif image_type == 'Streets':
        texture = extract_lbp(processed_image)
        gradients = extract_hog(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([texture, gradients, color_hist])
    return combined_features

def load_and_preprocess_images(folder, feature_set_extractor):
    features = []
    labels = []
    max_length = 0  # Initialize maximum length to zero
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                img_path = os.path.join(category_path, filename)
                image = cv.imread(img_path)
                if image is not None:
                    feature_vector = feature_set_extractor(image, category)
                    features.append(feature_vector)
                    labels.append(category)
                    max_length = max(max_length, len(feature_vector))  # Update maximum length
    
    # Pad feature vectors to the length of the longest vector
    for i in range(len(features)):
        features[i] = np.pad(features[i], (0, max_length - len(features[i])), mode='constant')
    return np.array(features), np.array(labels)

folder_path = 'dataset/dataset_full'
features_set3, labels_set3 = load_and_preprocess_images(folder_path, extract_features_set3)

import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def train_and_evaluate(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Save the scaler
    joblib.dump(scaler, 'scaler(03_02).joblib')
    
    # Train SVM model
    svm = SVC(C=1.0, kernel='linear', gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"SVM - Training Accuracy: {train_accuracy:.4f}")
    print(f"SVM - Test Accuracy: {test_accuracy:.4f}")
    print("SVM Classification Report:\n", classification_report(y_test, y_pred_test))
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.bar(['Training Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'orange']) # type: ignore
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('SVM: Training vs Test Accuracy')
    plt.grid(True)
    plt.show()

    # Save the trained model
    model_path='svm_model_set3.joblib'
    joblib.dump(svm, model_path)



print("Feature Set 3:")
train_and_evaluate(features_set3, labels_set3)