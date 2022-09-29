import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm

from model import build_model
from datasets import get_data_loaders, get_datasets
from utils import save_model, save_plots

ROOT_DIR_TRAIN = '.../inputs/data/train'
ROOT_DIR_VALID = '.../inputs/data/val'

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 4

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=20,
    help='Number of epochs to train our network for'
)

parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.0001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        total += len(labels)
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()    
        # Schedule
        scheduler.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / total)
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1            
            image, labels = data
            total += len(labels)
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    # epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    epoch_acc = 100. * (valid_running_correct / total)
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(ROOT_DIR_TRAIN, ROOT_DIR_VALID)
    print(f'[INFO]: Number of training images: {len(dataset_train)}')
    print(f'[INFO]: Number of validation images: {len(dataset_valid)}')
    print(f'[INFO]: Class names: {dataset_classes}\n')
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
   
    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Computation device: {device}')
    print(f'Learning rate: {lr}')
    print(f'Epochs to train for: {epochs}\n')

    # Build model
    model = build_model(num_classes=len(dataset_classes)).to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Learning rate schedule
    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                    T_0=10, 
                                    T_mult=1, 
                                    eta_min=0.000001, 
                                    last_epoch=-1)
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    # Start the training.
    for epoch in range(epochs):
        print(f'[INFO]: Epoch {epoch+1} of {epochs}')

        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f'Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}')
        print(f'Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}')
        print('-'*50)

        time.sleep(5)
        
    # Save the trained model weights.
    save_model(epoch, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
            
    print('TRAINING COMPLETE')
    print(f'Training acc: {sum(train_acc) / len(train_acc):.3f}')
    print(f'Training acc: {sum(valid_acc) / len(valid_acc):.3f}')
