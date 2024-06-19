import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time


# Function to measure time
def measure_time(start_time):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


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


# Function to load an image using PIL and return a Tkinter-compatible image
def load_image(img_path, size=(150, 150)):
    img = Image.open(img_path)
    img = img.resize(size, Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    return img_tk


# Function to select an input image and display top similar images
def select_image():
    input_img_path = filedialog.askopenfilename()
    if input_img_path:
        top_similar_paths, top_similar_scores = get_top_n_similar_images(input_img_path, base_model, database_features, database_paths)

        # Display input image
        input_img = load_image(input_img_path)
        input_img_label.config(image=input_img)
        input_img_label.image = input_img

        # Display similar images
        for i in range(5):
            similar_img = load_image(top_similar_paths[i])
            similar_img_labels[i].config(image=similar_img)
            similar_img_labels[i].image = similar_img
            similar_img_scores[i].config(text=f"Score: {top_similar_scores[i]:.2f}")


# Create main window
root = tk.Tk()
root.title("Image Similarity Search")
root.geometry("1200x800")

# Input image label
input_img_label = tk.Label(root)
input_img_label.pack()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

select_img_btn = tk.Button(btn_frame, text="Select Image", command=select_image)
select_img_btn.pack(side=tk.LEFT, padx=10)

# Similar images labels
similar_imgs_frame = tk.Frame(root)
similar_imgs_frame.pack()

similar_img_labels = []
similar_img_scores = []
for i in range(5):
    frame = tk.Frame(similar_imgs_frame)
    frame.pack(side=tk.LEFT, padx=10)
    img_label = tk.Label(frame)
    img_label.pack()
    score_label = tk.Label(frame, text="Score: ")
    score_label.pack()
    similar_img_labels.append(img_label)
    similar_img_scores.append(score_label)

# Start main loop
root.mainloop()
