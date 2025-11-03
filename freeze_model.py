import torch
from collections import OrderedDict
import sys
import timm
from pathlib import Path

# --- CONFIGURATION ---
# 1. Path to your *original* training checkpoint
ORIGINAL_CHECKPOINT_PATH = "checkpoints/accuracy=0.774.ckpt" 

# 2. Path to save the *new* "frozen" model for the app
FROZEN_MODEL_PATH = "models/resnet50_imagenet_frozen.pth"
# ---------------------

def freeze_checkpoint(original_path, frozen_path):
    """
    Loads a PyTorch training checkpoint, extracts the model's state_dict,
    cleans it (e.g., removes 'module.' prefix from DataParallel),
    and saves the clean state_dict for inference.
    """
    print(f"Loading checkpoint from: {original_path}")
    try:
        # Load checkpoint onto CPU
        checkpoint = torch.load(original_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {original_path}")
        print("Please update the ORIGINAL_CHECKPOINT_PATH variable in this script.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # --- Extract the state_dict ---
    # Checkpoints can be saved in different ways.
    # We try to find the actual model weights.
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assuming the checkpoint *is* the state_dict
        state_dict = checkpoint
        print("Note: Checkpoint seems to be a raw state_dict.")

    # --- Clean the state_dict keys ---
    # Lightning saves model weights with 'model.' prefix
    # DataParallel saves with 'module.' prefix
    # We need to remove these for loading into a standard model.
    cleaned_state_dict = OrderedDict()
    keys_cleaned = 0
    for k, v in state_dict.items():
        name = k
        # Remove 'model.' prefix (Lightning)
        if name.startswith('model.'):
            name = name[6:]  # remove 'model.'
            keys_cleaned += 1
        # Remove 'module.' prefix (DataParallel)
        elif name.startswith('module.'):
            name = name[7:]  # remove 'module.'
            keys_cleaned += 1
        cleaned_state_dict[name] = v
    
    if keys_cleaned > 0:
        print(f"Cleaned {keys_cleaned} keys (removed Lightning/DataParallel prefixes).")

    # --- Get num_classes from checkpoint ---
    num_classes = 1000  # Default for ImageNet-1k
    if 'hyper_parameters' in checkpoint:
        num_classes = checkpoint['hyper_parameters'].get('num_classes', 1000)
        print(f"Found num_classes in checkpoint: {num_classes}")
    
    # --- Verify and Save the "Frozen" Model ---
    try:
        # 1. Create a ResNet-50d instance with correct num_classes
        # resnet50d is the architecture you trained with
        print(f"Creating resnet50d model with {num_classes} classes...")
        model = timm.create_model('resnet50d', pretrained=False, num_classes=num_classes)

        # 2. Load our cleaned state_dict
        print("Loading weights into model...")
        model.load_state_dict(cleaned_state_dict)
        print("✅ Weights loaded successfully!")
        
        # 3. Save *only* the state_dict. This is our "frozen" file.
        # Create output directory if it doesn't exist
        Path(frozen_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving frozen model to: {frozen_path}")
        torch.save(model.state_dict(), frozen_path)
        
        print(f"\nSuccess! Frozen model state_dict saved to: {frozen_path}")
        print("You can now upload this file to your Hugging Face Space.")
    
    except RuntimeError as e:
        print("\nError: A problem occurred while loading the state_dict.")
        print("This often happens if the model architecture doesn't match the weights.")
        print("Details:", e)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    freeze_checkpoint(ORIGINAL_CHECKPOINT_PATH, FROZEN_MODEL_PATH)

