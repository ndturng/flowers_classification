import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from repvgg import get_RepVGG_func_by_name
# from repvgg import create_RepVGG_A0

# from model import build_model
# from datasets import get_datasets, get_data_loaders
from datasets import get_train_transform, get_valid_transform, WrapperDataset
from utils import save_model, save_plots

ROOT_DIR = '.../inputs/data/train'
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
    kfold = KFold(n_splits=10, shuffle=True)
    dataset = datasets.ImageFolder(ROOT_DIR)
    dataset_classes = dataset.classes 

    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Computation device: {device}')
    print(f'Learning rate: {lr}')
    print(f'Epochs to train for: {epochs}\n')

    # model = build_model(
    #     pretrained=args['pretrained'], 
    #     fine_tune=True, 
    #     num_classes=len(dataset_classes)
    # ).to(device)
    
    repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A0')
    model = repvgg_build_func(deploy=False)
    model.to(device)

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

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    # Start the training.
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # print(fold, len(train_ids), len(test_ids))
        print(f'[INFO]: Fold {fold+1} of {fold}')


        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(
                        WrapperDataset(dataset, transform=get_train_transform(IMAGE_SIZE)),
                        batch_size=BATCH_SIZE, 
                        sampler=train_subsampler, 
                        num_workers=NUM_WORKERS)
        testloader = DataLoader(
                        WrapperDataset(dataset, transform=get_valid_transform(IMAGE_SIZE)),
                        batch_size=BATCH_SIZE, 
                        sampler=test_subsampler, 
                        num_workers=NUM_WORKERS)

        for epoch in range(epochs):
            print(f'[INFO]: Epoch {epoch+1} of {epochs}')

            train_epoch_loss, train_epoch_acc = train(model, trainloader, 
                                                    optimizer, criterion)
            valid_epoch_loss, valid_epoch_acc = validate(model, testloader,  
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
            save_model(epoch, model, optimizer, criterion, args['pretrained'])
            # Save the loss and accuracy plots.
            save_plots(train_acc, valid_acc, train_loss, valid_loss, args['pretrained'])
            
    print('TRAINING COMPLETE')
    print(f'Training acc: {sum(train_acc) / len(train_acc):.3f}')
    print(f'Training acc: {sum(valid_acc) / len(valid_acc):.3f}')
