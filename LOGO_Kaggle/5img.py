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


# Function to preprocess a single image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


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

# Check if features are cached, otherwise extract and cache them
cache_file = "database_features.pkl"

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        database_features, database_paths = pickle.load(f)
else:
    batch_size = 32
    database_features = []
    for i in range(0, len(database_paths), batch_size):
        batch_paths = database_paths[i:i + batch_size]
        batch_features = extract_features_batch(base_model, batch_paths)
        database_features.append(batch_features)
    database_features = np.vstack(database_features)
    with open(cache_file, 'wb') as f:
        pickle.dump((database_features, database_paths), f)

# Example ground truth pairs (you need to create this based on your dataset)
ground_truth = [
    (r"C:\Users\LENOVO\Documents\VSN LEARNING\download (3).jpeg",
     r"C:\Users\LENOVO\Documents\VSN LEARNING\Logo\correct_match.jpeg"),
    # Add more pairs here
]


# Function to get top N similar images
def get_top_n_similar_images(input_img_path, base_model, database_features, database_paths, top_n=5):
    input_features = extract_features_batch(base_model, [input_img_path])
    similarities = calculate_similarity_matrix(input_features, database_features)
    similarity_scores = similarities[0]  # Similarities for the input image

    # Get indices of top N similar images
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    top_similar_paths = [database_paths[idx] for idx in top_indices]
    top_similar_scores = [similarity_scores[idx] for idx in top_indices]

    return top_similar_paths, top_similar_scores


# Example usage for displaying top 5 similar images
input_img_path = r"C:\Users\LENOVO\Documents\VSN LEARNING\download (5).jpeg"
top_similar_paths, top_similar_scores = get_top_n_similar_images(input_img_path, base_model, database_features,
                                                                 database_paths)

# Display input image and top 5 most similar images
input_img = image.load_img(input_img_path)
plt.figure(figsize=(15, 7))

plt.subplot(2, 3, 1)
plt.imshow(input_img)
plt.title("Input Image")
plt.axis('off')

for i, (img_path, score) in enumerate(zip(top_similar_paths, top_similar_scores), 2):
    similar_img = image.load_img(img_path)
    plt.subplot(2, 3, i)
    plt.imshow(similar_img)
    plt.title(f"Similar Image {i - 1}\nScore: {score:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# End timer and measure time taken
measure_time(start_time)
