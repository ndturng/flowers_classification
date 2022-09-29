import torch
import cv2
import numpy as np
import glob as glob
import os
import torch.nn as nn


from repvgg import create_RepVGG_A0
from model import build_model
from torchvision import transforms

# Constants.
DATA_PATH = '.../inputs/data/test/tulip'
MODEL_PATH = '.../out_cosinwarm_scheduler/model_pretrained_50.pth'

IMAGE_SIZE = 128
DEVICE = 'cpu'

# Class names.
class_names = ['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']

# Load the trained model.

model = create_RepVGG_A0(deploy=False)
# Change the final layer
num_ftrs = model.linear.in_features
model.linear = nn.Linear(num_ftrs, 7)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Get all the test image paths.
all_image_paths = glob.glob(f'{DATA_PATH}/*')
# Iterate over all the images and do forward pass.
# len(all_image_paths)

for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-2]  #.split('.')[0]
    image_name = image_path.split(os.path.sep)[-1]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()
    
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)
    
    # Forward pass throught the image.
    outputs = model(image)
    outputs = outputs.detach().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    print(f"GT: {gt_class_name} - image name: {image_name} - Pred: {pred_class_name.lower()}")

