# app.py
from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import logging
import os # Added for path checking
import requests # Added for downloading labels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Load the pre-trained PyTorch model ---
# We'll use ResNet18, a popular and relatively small CNN model, pre-trained on ImageNet.
try:
    # Load a pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval() # Set the model to evaluation mode (important for inference)
    logger.info("ResNet18 model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ResNet18 model: {e}")
    model = None # Set to None if loading fails

# --- Image Preprocessing Transform ---
# Define the transformations required for the input image for ResNet18
# All pre-trained models expect input images normalized in a certain way.
preprocess = transforms.Compose([
    transforms.Resize(256),        # Resize the image to 256x256
    transforms.CenterCrop(224),    # Crop the center 224x224 pixels
    transforms.ToTensor(),         # Convert image to PyTorch Tensor
    transforms.Normalize(          # Normalize with ImageNet's mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --- Routes ---

# Route for the home page, serving the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to perform image classification
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Image classification model not loaded. Please check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file.'}), 400

    if file:
        try:
            # Read image bytes
            img_bytes = file.read()
            # Open image using PIL (Pillow)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB') # Ensure RGB format

            # Apply preprocessing transforms
            input_tensor = preprocess(img)
            # Add a batch dimension (C, H, W) -> (1, C, H, W)
            input_batch = input_tensor.unsqueeze(0)

            # Move input to CPU (if not already there)
            # For this demo, we assume CPU. For GPU, you'd add:
            # if torch.cuda.is_available():
            #     input_batch = input_batch.to('cuda')
            #     model.to('cuda')

            # Make prediction
            with torch.no_grad(): # Disable gradient calculation for inference
                output = model(input_batch)

            # Get probabilities and predicted class index
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)

            # Decode predictions
            results = []
            if not hasattr(app, 'imagenet_labels') or not app.imagenet_labels:
                logger.warning("ImageNet labels not loaded. Using generic labels (class index).")
                for i in range(top3_indices.size(0)):
                    idx = top3_indices[i].item()
                    score = top3_prob[i].item()
                    results.append({
                        'label': f"Class Index {idx}",
                        'score': round(score * 100, 2)
                    })
            else:
                for i in range(top3_indices.size(0)):
                    idx = top3_indices[i].item()
                    score = top3_prob[i].item()
                    label = app.imagenet_labels[idx] if idx < len(app.imagenet_labels) else f"Unknown Class {idx}"
                    results.append({
                        'label': label,
                        'score': round(score * 100, 2)
                    })

            logger.info(f"Image predicted: {results[0]['label']} with {results[0]['score']}% confidence.")
            return jsonify({'predictions': results})

        except Exception as e:
            logger.error(f"Error during image prediction: {e}")
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500
    
    return jsonify({'error': 'An unknown error occurred.'}), 500

# --- Function to load ImageNet labels ---
def load_imagenet_labels():
    labels = []
    try:
        # Try to load from a local file first
        local_path = 'imagenet_classes.txt'
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
            app.imagenet_labels = labels
            logger.info("ImageNet labels loaded successfully from local file.")
            return

        # Fallback to downloading if local file not found
        logger.info("Local 'imagenet_classes.txt' not found. Attempting to download from URL.")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        labels = [line.strip() for line in response.text.split('\n') if line.strip()]
        app.imagenet_labels = labels
        logger.info("ImageNet labels loaded successfully from URL.")
    except Exception as e:
        logger.error(f"Could not load ImageNet labels: {e}. Predictions will show class index.")
        app.imagenet_labels = [] # Fallback to empty list

# --- Run the Flask app ---
if __name__ == '__main__':
    # Load labels when the app starts
    load_imagenet_labels()
    app.run(debug=True, port=5000)
