
# End to End Image Classification Using K Nearest Neighbors

## Objective
Demonstrate proficiency in solving a computer vision problem using handcrafted features and shallow learning models. By Implementing various preprocessing techniques and extracting handcrafted features, candidates should gain insights into how different features contribute to image classification tasks. The assignment also includes developing a Flask application for image classification.

## Assignment Components 
Assignment Components ->
1. Data Downloading and preprocessing
2. Feature Extraction 

    - You need to tak multiple sets of handcrafted features, keep 3 features at last in a feature set.

    - The features can be Low-level Vision: Histogram and Histogram Equalization, gray-scale transformation, Image Smoothing, Connected Components in images. 

    - Mid-level Vision : Edge detection using Gradients, Sobel, Canny; Line detection using Hough transform; Semantic information using RANSAC; Image region descriptor using SIFT etc. 
3. Dimensionality reduction if needed
4. Classification algorithm of your choice with explanation
5. Evaluation components
6. Flask App: one should be able to upload an image and get the classification result.

## Deliverables

1. Explain image preprocessing steps.
2. Explain the importance of your selected feature sets for this image classification task.
3. Apply appropriate techniques for dimensionality reduction, if your feature set size is too large and explain that.
4. Evaluation of the trained models using appropriate metrics.
5. Comparison of results obtained from different feature sets.
6. Development of a Flask application with image upload functionality for classification from best performing model.
7. Guide on setting up a Flask application for local image classification.
8. Enhancement scope to improve the performance of the model, also is there any way we can automate the feature extraction process

## Note
   - Do not use CNN/RNN for this assignment
   - Push code to Github

##  Guide for Local Implementation of Project:

### Project Structure:

   1. app.py: Flask web application for image classification.
   2. index.html: HTML template for the web application interface.
   3. dataset/: Directory containing the image dataset.

   4. feature_extraction_and_pca.py: Script for feature extraction and PCA.

   5. train_and_save_model.py: Script for training and saving the KNN model.

   6. requirements.txt: File listing project dependencies.


### Prerequisites:
   - Python 3.x installed on your system.
   - Required Python packages installed (NumPy, OpenCV, scikit-learn, Matplotlib, Flask).

### Steps:
    
1. Clone the Repository:

            git clone <repository_url>

2. Navigate to Project Directory:

            cd end_to_end_image_classification

3. Install Dependencies:

            pip install -r requirements.txt

    4. Prepare Dataset:

        Place your image dataset in the dataset/dataset_full directory.

        Ensure the dataset contains subdirectories for each class/category.

    5. Run the train_and_save_model.py script.

        The file Includes code for feature extraction, model training and saving the model for deployment.

    6. Launch the Flask web application by running the app.py script.        

        Open the terminal, Click on the Link created.

    7. Upload Images for Classification:

        Visit the web application in your browser and upload images for classification.


