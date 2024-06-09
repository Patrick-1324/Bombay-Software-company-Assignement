import os
import joblib  # For saving and loading models
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# *************************************** BASIC IMAGE PROCESSING ************************************
def preprocess_image(image):
    resized_image = cv.resize(image, (150, 150))
    gray_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image
# *************************************** BASIC IMAGE PROCESSING ************************************


# *************************************** FEATURE DEFINITION ************************************
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
    return h.flatten()  # type: ignore

def extract_corners(image):
    corners = cv.cornerHarris(image, 2, 3, 0.04)
    return corners.flatten()
# *************************************** FEATURE DEFINITION ************************************


# ************************************ FEATURE EXTRACTION (SET1) ********************************
def extract_features_set1(image, image_type):
    processed_image = preprocess_image(image)
    color_image = cv.cvtColor(cv.resize(image, (150, 150)), cv.COLOR_BGR2RGB)
    combined_features = []
    if image_type == 'Buildings':
        edges = extract_sobel_edges(processed_image)
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([edges, texture, color_hist])
    elif image_type == 'Forest':
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([texture, color_hist])
    elif image_type == 'Glacier':
        edges = extract_sobel_edges(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([edges, color_hist])
    elif image_type == 'Mountains':
        edges = extract_sobel_edges(processed_image)
        texture = extract_lbp(processed_image)
        combined_features = np.hstack([edges, texture])
    elif image_type == 'Sea':
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([texture, color_hist])
    elif image_type == 'Streets':
        edges = extract_sobel_edges(processed_image)
        texture = extract_lbp(processed_image)
        color_hist = extract_histogram(color_image)
        combined_features = np.hstack([edges, texture, color_hist])
    return combined_features
# ************************************ FEATURE EXTRACTION (SET1) ********************************


# *********************************** DATA LOADING AND PROCESSING ********************************
def load_and_preprocess_images(folder, feature_set_extractor):
    print("Loading the Images /.")
    features = []
    labels = []
    max_length = 0  # Initialize maximum length to zero
    # First pass: Extract features and find the maximum feature vector length
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
    print("Data Padding (if needed) /.")
    # Second pass: Pad feature vectors to the length of the longest vector
    for i in range(len(features)):
        features[i] = np.pad(features[i], (0, max_length - len(features[i])), mode='constant')
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    # Convert to numpy array
    features = np.array(features)
    encoded_labels = np.array(encoded_labels)
    return features, encoded_labels, label_encoder
# *********************************** DATA LOADING AND PROCESSING ********************************


# ************************************** OPTIMAL PCA COMPONENT ***********************************
def select_optimal_pca_components(features, labels, variance_threshold=0.95):
    print("Optimal_PCA_calculations /.")
    print("     1. plotting Variance Ratio ->")
    # Step 1: Plot Explained Variance Ratio
    pca = PCA().fit(features)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.grid(True)
    plt.show()
    # Step 2: Determine n_components for desired variance threshold
    print("     2. n_Components for variance threshold  ->")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Number of components to retain {variance_threshold*100}% variance: {n_components}")
    # Step 3: Perform Cross-Validation
    print("     3. Plotting components vs accuracy  ->")
    n_components_range = range(10, n_components + 50, 10)  # Adjust range based on your data
    mean_accuracies = []
    for n in n_components_range:
        pca = PCA(n_components=n)
        features_reduced = pca.fit_transform(features)
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, features_reduced, labels, cv=5)  # 5-fold cross-validation
        mean_accuracies.append(scores.mean())
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, mean_accuracies)
    plt.xlabel('Number of Components')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Model Performance vs Number of Components')
    plt.grid(True)
    plt.show()
    optimal_n_components = n_components_range[np.argmax(mean_accuracies)]
    print(f"Optimal number of components: {optimal_n_components}")
    return optimal_n_components
# ************************************** OPTIMAL PCA COMPONENT ***********************************


# ****************************************** IMPLEMENTATION **************************************
folder_path = 'dataset/dataset_full'
features_set3, labels_set3, label_encoder = load_and_preprocess_images(folder_path, extract_features_set1)

# Select optimal n_components
optimal_n_components = select_optimal_pca_components(features_set3, labels_set3)
# Apply PCA with the optimal number of components
pca = PCA(n_components=optimal_n_components)
features_set3_reduced = pca.fit_transform(features_set3)
print("Original Features shape ->", features_set3.shape)
print("PCA Features shape ->", features_set3_reduced.shape)

# Save the PCA model, label encoder, and trained KNN model
joblib.dump(pca, 'pca_model_optimal.pkl')
joblib.dump(label_encoder, 'label_encoder_pca.pkl')
# ****************************************** IMPLEMENTATION **************************************


# ************************************** MODEL TRAINING AND EVAL *********************************
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
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, knn.predict(X_train))
    test_accuracy = accuracy_score(y_test, knn.predict(X_test))
    print(f"KNN with k= 5 - Training Accuracy: {train_accuracy:.4f}")
    print(f"KNN with k= 5 - Test Accuracy: {test_accuracy:.4f}")
    print("KNN Classification Report:\n", classification_report(y_test, knn.predict(X_test)))
    # Save the trained model
    joblib.dump(knn, model_path)
    print(f"Model saved to {model_path}")

# Train and save the model
model_path = 'knn_set3_pca.pkl'
print("Training and saving the model:")
train_and_save_model(features_set3_reduced, labels_set3, model_path)
# ************************************** MODEL TRAINING AND EVAL *********************************