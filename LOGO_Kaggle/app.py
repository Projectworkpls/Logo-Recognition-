import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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


# Load database features and paths
with open('database_features.pkl', 'rb') as f:
    database_features, database_paths = pickle.load(f)


# Tkinter GUI
class ImageSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Similarity Search")

        self.input_img_path = ""
        self.most_similar_img_path = ""
        self.similarity_score = 0.0

        # Create input section
        self.lbl_input = tk.Label(self.root, text="Upload an image:")
        self.lbl_input.pack(pady=10)

        self.btn_upload = tk.Button(self.root, text="Upload Image", command=self.upload_image, width=20, height=2)
        self.btn_upload.pack()

        self.input_img_label = tk.Label(self.root)
        self.input_img_label.pack(pady=10)

        # Create results section
        self.lbl_result = tk.Label(self.root, text="Most Similar Image:")
        self.lbl_result.pack()

        self.similar_img_label = tk.Label(self.root)
        self.similar_img_label.pack(pady=10)

        self.lbl_similarity = tk.Label(self.root, text="")
        self.lbl_similarity.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.input_img_path = file_path
            self.display_input_image()
            self.find_most_similar_image()

    def display_input_image(self):
        img = Image.open(self.input_img_path)
        img = img.resize((300, 300), Image.LANCZOS)  # Resize input image
        img = ImageTk.PhotoImage(img)

        self.input_img_label.configure(image=img)
        self.input_img_label.image = img

    def find_most_similar_image(self):
        try:
            input_features = extract_features(base_model, self.input_img_path)
            similarities = cosine_similarity(input_features, database_features)
            best_similarity_idx = np.argmax(similarities)
            self.most_similar_img_path = database_paths[best_similarity_idx]
            self.similarity_score = similarities[0, best_similarity_idx]

            self.display_most_similar_image()
            self.display_similarity_score()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_most_similar_image(self):
        img = Image.open(self.most_similar_img_path)
        img = img.resize((300, 300), Image.LANCZOS)  # Resize similar image
        img = ImageTk.PhotoImage(img)

        self.similar_img_label.configure(image=img)
        self.similar_img_label.image = img

    def display_similarity_score(self):
        self.lbl_similarity.config(text=f"Similarity: {self.similarity_score * 100:.2f}%")


if __name__ == "__main__":
    # Initialize Tkinter
    root = tk.Tk()
    app = ImageSimilarityApp(root)
    root.mainloop()
