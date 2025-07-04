# AI Image Classifier (Flask & PyTorch)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/AI-Computer_Vision-green?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-FF2D20?style=for-the-badge&logo=tensorflow&logoColor=white)
![Fullstack AI](https://img.shields.io/badge/AI_Application-Fullstack-blueviolet?style=for-the-badge)

### Project Overview

This project is a web-based application that allows users to upload an image and receive a classification prediction (identifying what the image contains) using a pre-trained deep learning model. It serves as a practical demonstration of applying Artificial Intelligence in Computer Vision and the deployment of a machine learning model within a web environment using Python Flask and PyTorch.

## Features

* **Image Upload & Preview:** Provides an intuitive drag-and-drop or click-to-upload interface for image selection, with an immediate visual preview of the selected image.
* **AI-Powered Classification:**
    * Utilizes a pre-trained **PyTorch ResNet18 model** (trained on the vast ImageNet dataset) for robust and accurate image classification.
    * Performs all necessary image preprocessing steps (resizing, cropping, normalization) to prepare images for the model's input requirements.
    * Predicts the top 3 most likely categories for the uploaded image.
* **Human-Readable Labels:** Displays meaningful, human-readable labels for the predictions (e.g., "sports car", "cat", "coffee mug") by loading ImageNet class names, making the results easily understandable.
* **Interactive Results Display:**
    * Shows prediction labels and confidence scores in a clear format.
    * Visualizes prediction confidence using an animating **progress bar** for each prediction.
    * Includes clear loading and error messages with visual cues.
* **Responsive UI:** The entire web interface is designed to adapt beautifully to various screen sizes, ensuring optimal usability on desktops, tablets, and mobile devices.
* **Full-Stack AI Application:** Combines a Python Flask backend (responsible for hosting the ML model and performing inference) with a simple HTML/CSS/JavaScript frontend (for handling user interaction and displaying results).

## Technologies Used

* **Python 3.x:** Core language for the backend logic and ML operations.
* **Flask:** A lightweight Python web framework used for serving the application and exposing the prediction API endpoint.
* **PyTorch:** A leading open-source machine learning framework used for loading and running the pre-trained deep learning model.
* **`torchvision`:** A PyTorch library providing access to popular datasets, pre-trained models (like ResNet18), and image transformations for computer vision.
* **`Pillow` (PIL):** The Python Imaging Library, essential for image loading, manipulation, and resizing.
* **HTML5, CSS3, JavaScript:** Used for developing the interactive and responsive frontend user interface.
* **`fetch` API:** For asynchronous communication between the frontend and the Flask backend API.
* **Font Awesome:** Integrated for scalable vector icons, enhancing the visual appeal of the UI.

### How to Download and Run the Project

### 1. Prerequisites

* **Python 3.x:** Ensure Python 3.x is installed on your system. Download from [python.org](https://www.python.org/downloads/).
* **`pip`:** Python's package installer.
* **Git:** Ensure Git is installed on your system. Download from [git-scm.com](https://git-scm.com/downloads/).
* **VS Code (Recommended):** For a smooth development experience.

### 2. Download the Project

1.  **Open your terminal or Git Bash.**
2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)AtharvaMeherkar/AI-Image-Classifier.git
    ```
3.  **Navigate into the project directory:**
    ```bash
    cd AI-Image-Classifier
    ```

### 3. Setup and Installation

1.  **Open the project in VS Code:**
    ```bash
    code .
    ```
2.  **Open the Integrated Terminal in VS Code** (`Ctrl + ~`).
3.  **Create and activate a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
    You should see `(venv)` at the beginning of your terminal prompt.
4.  **Install the required Python packages:**
    ```bash
    pip install Flask torch torchvision Pillow requests
    ```
    *(Note: `torch` and `torchvision` are large downloads (hundreds of MBs) and may take several minutes depending on your internet connection. Please be patient.)*
5.  **Download ImageNet Labels:**
    * The `app.py` script attempts to download `imagenet_classes.txt` from GitHub on startup. However, if there are network issues, you can download it manually:
    * Download `imagenet_classes.txt` from: `https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt`
    * Save this file directly into the `AI-Image-Classifier` project folder (the same folder as `app.py`). This ensures human-readable labels for predictions.

### 4. Execution

1.  **Ensure your virtual environment is active** in the VS Code terminal.
2.  **Set the Flask application environment variable:**
    ```bash
    # On Windows:
    $env:FLASK_APP = "app.py"
    # On macOS/Linux:
    export FLASK_APP=app.py
    ```
3.  **Run the Flask development server:**
    ```bash
    python -m flask run
    ```
    *(The first time you run this, the PyTorch model weights will be downloaded, which can take a few minutes. Wait for this to complete. You'll see progress bars in your terminal.)*
4.  **Open your web browser** and go to `http://127.0.0.1:5000` (or `http://localhost:5000`).
5.  **Upload an image:** Drag and drop or click to select an image file (e.g., a photo of a cat, dog, car, airplane, common household objects work best as the model is trained on ImageNet).
6.  **Click "Predict Image"** and observe the predictions with labels, confidence scores, and animated progress bars.


## What I Learned / Challenges Faced

* **Applied Computer Vision with PyTorch:** Gained hands-on experience in loading, preprocessing, and performing inference with pre-trained deep learning models (ResNet18) for image classification.
* **Transfer Learning:** Understood the powerful concept of leveraging existing large models trained on vast datasets for new, specific tasks.
* **Full-Stack AI Deployment:** Learned to integrate a machine learning model into a functional web application, encompassing both backend (Flask for model hosting and API) and frontend (HTML/CSS/JavaScript for user interaction).
* **Image Preprocessing Pipeline:** Implemented the crucial steps required to prepare raw image data (resizing, cropping, normalization) to meet the input requirements of a neural network.
* **User Experience for AI Tools:** Focused on creating an intuitive and visually engaging interface for AI interaction, including clear loading states, dynamic image previews, and interactive result visualization.
* **Managing ML Dependencies:** Gained experience in handling the installation, downloading, and loading of significant model files and associated libraries like Hugging Face Transformers.
