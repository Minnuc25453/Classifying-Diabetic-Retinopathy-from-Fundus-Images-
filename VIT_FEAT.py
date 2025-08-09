import numpy as np
from transformers import ViTModel, ViTImageProcessor
import torch


def VIT(Imges):
    # Load the ViT model
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    # Load the image processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    # Load the image
    image = image_processor(Imges, return_tensors="pt")
    # Extract features
    with torch.no_grad():
        features = model(image.pixel_values).last_hidden_state
    features = np.asarray(features)
    return features
