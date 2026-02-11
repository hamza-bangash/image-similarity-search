from PIL import Image
from torchvision import transforms

image_size = 224

#### image preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def preprocess_image(image:Image.Image):
    image = image.convert("RGB")
    tensor = transform(image)
    return tensor
