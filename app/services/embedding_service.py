import torch
import numpy as np
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad=False

    return model

def extract_embedding(model,image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)

    embedding = embedding.squeeze().cpu().numpy()

    return embedding
