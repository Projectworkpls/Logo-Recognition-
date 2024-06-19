# Logo-Recognition-



The Logo or image Similarity App is a Python application that allows users to find the top 5 most similar images from a predefined database, given an input image. It uses the ResNet50 model for feature extraction and cosine similarity for comparing images based on their feature vectors.

Features
Browse Image: Allows users to select an image file from their local system.
Display Similar Images: Displays the top 5 most similar images from the database based on the selected input image.
Cosine Similarity: Calculates the similarity between images using cosine similarity metric.
Tkinter GUI: Provides a user-friendly interface to interact with the application.
Installation
Prerequisites
Python 3.x
TensorFlow
Tkinter (usually comes pre-installed with Python)

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Replace requirements.txt with a file containing necessary dependencies like tensorflow, numpy, scikit-learn, and Pillow.

Usage
Run the application:

bash
Copy code
python 5front.py
The application window will appear.

Click on the Browse Image button to select an image file from your local system.

Once the image is selected, click on the Display Similar Images button to see the top 5 most similar images from the database.

Close the application window when finished.

How It Works
Feature Extraction: Uses the ResNet50 model to extract feature vectors from images.
Cosine Similarity: Computes cosine similarity scores between the feature vectors of the input image and images in the database.
Tkinter GUI: Provides a graphical user interface for easy interaction.
Directory Structure
bash
Copy code
├── README.md                 # This file
├── 5front.py                 # Main application script
├── database_features.pkl     # Cached feature vectors of database images
├── requirements.txt          # List of dependencies
└── your-database-folder/     # Directory containing your database images
    ├── image1.jpg
    ├── image2.png
    └── ...
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
