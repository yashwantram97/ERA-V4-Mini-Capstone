import gradio as gr
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# --- CONFIGURATION ---
MODEL_PATH = "models/resnet50_imagenet_frozen.pth"
LABELS_PATH = "imagenet_labels.txt"
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# ---------------------

def download_labels(url, path):
    """Downloads the ImageNet labels file if it doesn't exist."""
    if not os.path.exists(path):
        print(f"Downloading labels from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, 'w') as f:
                f.write(response.text)
            print("Labels downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading labels: {e}")
            return None
    
    # Load labels from the file
    try:
        with open(path, 'r') as f:
            # We strip commas and quotes, as the file has them
            labels = [line.strip().split("', '")[0].replace("'", "").replace(",", "") for line in f.readlines()]
        return labels
    except FileNotFoundError:
        print(f"Labels file not found at {path} and download failed.")
        return None
    except Exception as e:
        print(f"Error reading labels file: {e}")
        return None

def load_model(model_path):
    """Loads the "frozen" ResNet-50 model."""
    print(f"Loading model from {model_path}...")
    try:
        # 1. Create the model architecture - try different variants
        # Try standard resnet50 first
        try:
            model = timm.create_model('resnet50', pretrained=False, num_classes=1000)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except RuntimeError as e:
            # If that fails, try resnet50d
            print("Standard resnet50 failed, trying resnet50d...")
            model = timm.create_model('resnet50d', pretrained=False, num_classes=1000)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        
        # 3. Set to evaluation mode
        model.eval()        
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Did you upload the 'resnet50_imagenet_frozen.pth' file?")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Main Setup ---
labels = download_labels(LABELS_URL, LABELS_PATH)
model = load_model(MODEL_PATH)

# Define the image transformations (standard for ImageNet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# ---------------------

def predict(image):
    """
    Prediction function that takes a PIL image and returns a
    dictionary of the top 5 predictions.
    """
    if model is None or labels is None:
        return {"Error": "Model or labels failed to load. Check the logs."}
        
    try:
        # 1. Preprocess the image
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)  # Create a mini-batch
        
        # 2. Run inference
        with torch.no_grad():
            output = model(batch_t)
        
        # 3. Get probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 4. Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        # 5. Format the results
        results = {}
        for i in range(top5_prob.size(0)):
            category_name = labels[top5_catid[i]].title()
            probability = top5_prob[i].item()
            results[category_name] = probability
            
        return results

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"Error": str(e)}

# --- Create and Launch the Gradio App ---
title = "ResNet-50 ImageNet Classifier"
description = ("This is a demo of a custom-trained ResNet-50 model "
               "for ImageNet-1k classification, deployed as a Hugging Face Space.")
article = ("<p style='text-align: center;'>Upload an image to see the model's top 5 predictions. For more examples visit the <a href='https://www.kaggle.com/datasets/mayurmadnani/imagenet-dataset?select=test'>ImageNet Test Examples</a></p>")

# We use gr.Label to get a nice output format for the {label: probability} dict
output_component = gr.Label(num_top_classes=5, label="Top 5 Predictions")
input_component = gr.Image(type="pil", label="Upload Image")

demo = gr.Interface(
    fn=predict,
    inputs=input_component,
    outputs=output_component,
    title=title,
    description=description,
    article=article,
    examples=[
        ["examples/ILSVRC2012_test_00000002.jpeg"],
        ["examples/ILSVRC2012_test_00000004.jpeg"],
        ["examples/ILSVRC2012_test_00000005.jpeg"],
        ["examples/ILSVRC2012_test_00000017.jpeg"],
        ["examples/ILSVRC2012_test_00000018.jpeg"],
        ["examples/ILSVRC2012_test_00000028.jpeg"],
        ["examples/ILSVRC2012_test_00000031.jpeg"],
    ]
)

if __name__ == "__main__":
    if model is None or labels is None:
        print("\n--- GRADIORUNTIME ERROR ---")
        print("The app cannot start because the model or labels failed to load.")
        print("Please check the error messages above.")
        print("If running on a Hugging Face Space, check the 'Files' tab")
        print("to ensure your 'resnet50_imagenet_frozen.pth' file is present.")
        print("---------------------------\n")
    else:
        print("Launching Gradio app...")
        demo.launch()
