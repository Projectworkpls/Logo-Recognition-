import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import time

# Function to measure time
def measure_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

# Start timer
start_time = time.time()

# Load ResNet50 model for feature extraction
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Function to preprocess a batch of images
def load_and_preprocess_images(img_paths):
    imgs = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        imgs.append(img_array)
    imgs_array = np.array(imgs)
    imgs_array = preprocess_input(imgs_array)
    return imgs_array

# Function to extract features for a batch of images
def extract_features_batch(model, img_paths):
    imgs_array = load_and_preprocess_images(img_paths)
    features = model.predict(imgs_array)
    return features

# Function to calculate similarity matrix between two sets of feature vectors
def calculate_similarity_matrix(features1, features2):
    similarities = cosine_similarity(features1, features2)
    return similarities

# Directory containing the database images
database_dir = r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo"

# Get all image paths in the database
database_paths = []
for root, dirs, files in os.walk(database_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            database_paths.append(os.path.join(root, file))

# Placeholder Vienna Classification data
# In practice, this should be a dictionary where keys are image paths and values are Vienna Classification codes.
vienna_classification = {
    r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo\image1.jpg": "5.3.11",
    r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo\image2.jpg": "3.1.10",
    # Add more classifications here
}

# Example input Vienna Classification code
input_vienna_code = "3.1.10"

# Filter database paths based on Vienna Classification
filtered_database_paths = [path for path in database_paths if vienna_classification.get(path) == input_vienna_code]

# Check if features are cached, otherwise extract and cache them
cache_file = "database_features.pkl"

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        database_features, filtered_database_paths = pickle.load(f)
else:
    batch_size = 32
    database_features = []
    for i in range(0, len(filtered_database_paths), batch_size):
        batch_paths = filtered_database_paths[i:i+batch_size]
        batch_features = extract_features_batch(base_model, batch_paths)
        database_features.append(batch_features)
    database_features = np.vstack(database_features)
    with open(cache_file, 'wb') as f:
        pickle.dump((database_features, filtered_database_paths), f)

# Example ground truth pairs (you need to create this based on your dataset)
ground_truth = [
    (r"C:\Users\LENOVO\Documents\VSN LEARNING\download (3).jpeg", r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo\correct_match.jpeg"),
    # Add more pairs here
]

# Calculate accuracy
correct_matches = 0
for input_img_path, correct_img_path in ground_truth:
    input_features = extract_features_batch(base_model, [input_img_path])
    similarities = calculate_similarity_matrix(input_features, database_features)
    best_similarity_idxs = np.argmax(similarities, axis=1)

    if best_similarity_idxs.size > 0:
        best_similarity_idx = best_similarity_idxs[0]
        if best_similarity_idx < len(filtered_database_paths):
            predicted_img_path = filtered_database_paths[best_similarity_idx]
            if predicted_img_path == correct_img_path:
                correct_matches += 1

accuracy = (correct_matches / len(ground_truth)) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Display the input image and the most similar image for a single test case
input_img_path = r"C:\Users\LENOVO\Documents\VSN LEARNING\download (4).jpeg"
input_features = extract_features_batch(base_model, [input_img_path])
similarities = calculate_similarity_matrix(input_features, database_features)
best_similarity_idx = np.argmax(similarities, axis=1)[0]
most_similar_image_path = filtered_database_paths[best_similarity_idx]
print(f"Most similar image path: {most_similar_image_path}")
print(f"Similarity: {similarities[0, best_similarity_idx] * 100:.2f}%")

input_img = image.load_img(input_img_path)
most_similar_img = image.load_img(most_similar_image_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(most_similar_img)
plt.title("Most Similar Image")
plt.axis('off')

plt.show()

# End timer and measure time taken
measure_time(start_time)
