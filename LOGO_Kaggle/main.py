import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load ResNet50 model for feature extraction
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Function to preprocess a single image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features from an image
def extract_features(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return features

# Function to calculate similarity between two feature vectors
def calculate_similarity(features1, features2):
    similarity = cosine_similarity(features1, features2)[0][0]
    return similarity

# Function to find the most similar image in the database
def find_most_similar_image(model, input_img_path, database_paths):
    input_features = extract_features(model, input_img_path)

    best_similarity = -1
    best_image_path = None

    for db_img_path in database_paths:
        db_features = extract_features(model, db_img_path)
        similarity = calculate_similarity(input_features, db_features)

        if similarity > best_similarity:
            best_similarity = similarity
            best_image_path = db_img_path

    return best_image_path, best_similarity

# Directory containing the database images
database_dir = r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo"

# Get all image paths in the database
database_paths = []
for root, dirs, files in os.walk(database_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            database_paths.append(os.path.join(root, file))

# Example ground truth pairs (you need to create this based on your dataset)
ground_truth = [
    (r"C:\Users\LENOVO\Documents\VSN LEARNING\download (3).jpeg", r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo\correct_match.jpeg"),
    # Add more pairs here
]

# Calculate accuracy
correct_matches = 0
for input_img_path, correct_img_path in ground_truth:
    predicted_img_path, _ = find_most_similar_image(base_model, input_img_path, database_paths)
    if predicted_img_path == correct_img_path:
        correct_matches += 1

accuracy = correct_matches / len(ground_truth)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the input image and the most similar image for a single test case
input_img_path = r"C:\Users\LENOVO\Documents\VSN LEARNING\download.png"
most_similar_image_path, similarity = find_most_similar_image(base_model, input_img_path, database_paths)
print(f"Most similar image path: {most_similar_image_path}")
print(f"Similarity: {similarity * 100:.2f}%")

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
