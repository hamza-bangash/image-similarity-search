import os
import json
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_DIR = 'data/Images'
OUtPUT_DIR = 'data/embeddings'

embeddings_file = os.path.join(OUtPUT_DIR,'embeddings.npy')
ids_file = os.path.join(OUtPUT_DIR,'images_ids.json')

image_size = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'



#### image preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


### loading resent50 
def load_model():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity() #### removing classifier
    model.eval()
    model.to(device)

    for param in model.parameters(): ### making model freze
        param.requires_grad = False 

    return model

##### loading a single image
def load_image(path):
    image = Image.open(path).convert("RGB") ## loading image
    image = transform(image)  ## applying preprocess
    return image


######## main engine which is response for embeddings of images and save the results

def generate_embeddings():
    model = load_model()   ### calling function to load clean and process model

    embeddings = list()
    image_ids = list()

    image_files = sorted(os.listdir(IMAGE_DIR)) ### read image name in sorted order

    for img_name in tqdm(image_files):

        img_path = os.path.join(IMAGE_DIR,img_name)

        image = load_image(img_path) #### calling fuction to load image and return clean image by applying transforms

        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image)

        embedding = embedding.squeeze().cpu().numpy()

        embeddings.append(embedding)
        image_ids.append(img_name)

    embeddings = np.vstack(embeddings)

    np.save(embeddings_file,embeddings)

    with open(ids_file,'w') as f:
        json.dump(image_ids,f)

    print('\n completed')

if __name__ == '__main__':
        generate_embeddings()