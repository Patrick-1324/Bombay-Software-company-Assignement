import os
import joblib  # For saving and loading models
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


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

# def extract_features_set1(image, image_type):
#     processed_image = preprocess_image(image)
#     color_image = cv.cvtColor(cv.resize(image, (150, 150)), cv.COLOR_BGR2RGB)
#     combined_features = []
#     if image_type == 'Buildings':
#         edges = extract_sobel_edges(processed_image)
#         texture = extract_lbp(processed_image)
#         color_hist = extract_histogram(color_image)
#         combined_features = np.hstack([edges, texture, color_hist])
#     elif image_type == 'Forest':
#         texture = extract_lbp(processed_image)
#         color_hist = extract_histogram(color_image)
#         combined_features = np.hstack([texture, color_hist])
#     elif image_type == 'Glacier':
#         edges = extract_sobel_edges(processed_image)
#         color_hist = extract_histogram(color_image)
#         combined_features = np.hstack([edges, color_hist])
#     elif image_type == 'Mountains':
#         edges = extract_sobel_edges(processed_image)
#         texture = extract_lbp(processed_image)
#         combined_features = np.hstack([edges, texture])
#     elif image_type == 'Sea':
#         texture = extract_lbp(processed_image)
#         color_hist = extract_histogram(color_image)
#         combined_features = np.hstack([texture, color_hist])
#     elif image_type == 'Streets':
#         edges = extract_sobel_edges(processed_image)
#         texture = extract_lbp(processed_image)
#         color_hist = extract_histogram(color_image)
#         combined_features = np.hstack([edges, texture, color_hist])
#     return combined_features

# def extract_features_set2(image, image_type):
#     processed_image = preprocess_image(image)
#     combined_features = []
#     if image_type == 'Buildings':
#         edges = extract_sobel_edges(processed_image)
#         corners = extract_corners(processed_image)
#         combined_features = np.hstack([edges, corners])
#     elif image_type == 'Forest':
#         edges = extract_sobel_edges(processed_image)
#         entropy = extract_entropy(processed_image)
#         combined_features = np.hstack([edges, entropy])
#     elif image_type == 'Glacier':
#         edges = extract_sobel_edges(processed_image)
#         entropy = extract_entropy(processed_image)
#         combined_features = np.hstack([edges, entropy])
#     elif image_type == 'Mountains':
#         edges = extract_sobel_edges(processed_image)
#         entropy = extract_entropy(processed_image)
#         combined_features = np.hstack([edges, entropy])
#     elif image_type == 'Sea':
#         edges = extract_sobel_edges(processed_image)
#         entropy = extract_entropy(processed_image)
#         combined_features = np.hstack([edges, entropy])
#     elif image_type == 'Streets':
#         edges = extract_sobel_edges(processed_image)
#         corners = extract_corners(processed_image)
#         combined_features = np.hstack([edges, corners])
#     return combined_features


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
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Pad feature vectors to the length of the longest vector
    for i in range(len(features)):
        features[i] = np.pad(features[i], (0, max_length - len(features[i])), mode='constant')

    # Save the label encoder for later use in the Flask app
    joblib.dump(label_encoder, 'label_encoder2.pkl')
    
    return np.array(features), np.array(encoded_labels)

folder_path = 'dataset/dataset_full'
features_set3, labels_set3 = load_and_preprocess_images(folder_path, extract_features_set3)


def train_and_save_model(features, labels, model_path):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Evaluate KNN for different values of k
    k_values = range(1, 21)  # Test k from 1 to 20
    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        # Calculate training accuracy
        train_accuracy = accuracy_score(y_train, knn.predict(X_train))
        train_accuracies.append(train_accuracy)
        # Calculate test accuracy
        test_accuracy = accuracy_score(y_test, knn.predict(X_test))
        test_accuracies.append(test_accuracy)
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(k_values, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('k-NN: Number of Neighbors vs. Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final evaluation with k=4 (as in the original code)
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, knn.predict(X_train))
    test_accuracy = accuracy_score(y_test, knn.predict(X_test))
    print(f"KNN with k=4 - Training Accuracy: {train_accuracy:.4f}")
    print(f"KNN with k=4 - Test Accuracy: {test_accuracy:.4f}")
    print("KNN Classification Report:\n", classification_report(y_test, knn.predict(X_test)))

    # Save the trained model
    joblib.dump(knn, model_path)
    print(f"Model saved to {model_path}")


# Train and save the model
model_path = 'knn_set3.pkl'
print("Training and saving the model for Feature Set 3:")
train_and_save_model(features_set3, labels_set3, model_path)


def load_and_predict(model_path, unseen_image_path, feature_extractor, feature_shape):
    knn = joblib.load(model_path)
    image = cv.imread(unseen_image_path)
    if image is not None:
        unseen_features = feature_extractor(image, None)
        unseen_features = np.pad(unseen_features, (0, feature_shape - len(unseen_features)), mode='constant')
        unseen_features = unseen_features.reshape(1, -1)  # Reshape for a single prediction
        unseen_prediction = knn.predict(unseen_features)
        print(f"Unseen Image: {unseen_image_path}, Predicted Category: {unseen_prediction[0]}")


# Predict on multiple unseen images
unseen_image_paths = ['unseen_data/Building_2.jpg', 'unseen_data/forest4.jpg', 'unseen_data/forest5.jpg',
                      'unseen_data/forest6.jpg', 'unseen_data/forest7.jpg', 'unseen_data/glacier_4.jpg',
                      'unseen_data/glacier_5.jpg', 'unseen_data/glacier_6.jpg', 'unseen_data/glacier_3.jpg',
                      'unseen_data/mountains_5.jpg', 'unseen_data/mountains_4.jpg', 'unseen_data/mountains_3.jpg',
                      'unseen_data/Mountain3.jpg', 'unseen_data/sea5.jpg', 'unseen_data/sea4.jpg', 'unseen_data/sea3.jpg',
                      'unseen_data/street.jpg', 'unseen_data/street_2.jpg']

for unseen_image_path in unseen_image_paths:
    print(f"Predicting for unseen image: {unseen_image_path}")
    load_and_predict(model_path, unseen_image_path, extract_features_set3, features_set3.shape[1])