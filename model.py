import torch.nn as nn
from repvgg import create_RepVGG_A0
from utils import load_checkpoint

def build_model(num_classes):

    print('[INFO]: Loading pre-trained weights')
    model = create_RepVGG_A0(deploy=False)
    # Load pretrained weight
    load_checkpoint(model, '.../RepVGG-A0-train.pth')
    # Freezing layer
    for params in model.parameters():
        params.requires_grad = False

    # Change the final layer
    num_ftrs = model.linear.in_features
    model.linear = nn.Linear(num_ftrs, num_classes)

    return model
