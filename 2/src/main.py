#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import os, time, copy, argparse
import numpy as np
from utils import *

from ParamLearner import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d','--data', metavar='DIR', type=str, default='../data',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('-k', default=1, type=int,
                    metavar='K', help='KNN param')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--step_size', default=7, type=int,
                    metavar='N', help='Decay LR by a factor of 0.1 every N epochs')
parser.add_argument('--prefix', type=str, metavar='MODEL_NAME', default='',
                    help='save trained model with this name')
parser.add_argument('--arch', type=str, metavar='MODEL_ARCH', default='parampred',
                    help='model arch: (baseline, parampred, knn)')
parser.add_argument('--ckpt', type=str, metavar='MODEL',
                    help='checkpoint name')
parser.add_argument('--drop_rate', default=0.5, type=float,
                    metavar='dp', help='Dropout rate')
parser.add_argument('--use_special_loader', dest='special_loader', action='store_true')
parser.add_argument('--no_use_special_loader', dest='special_loader', action='store_false')
parser.set_defaults(special_loader=True)

# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
args = parser.parse_args()

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# Data augmentation and normalization for training
# Just normalization for validation

train_arg = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
val_arg = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = {
    'train': transforms.Compose(train_arg if args.arch != 'knn' else val_arg),
    'val': transforms.Compose(val_arg),
    'stable_train': transforms.Compose(val_arg)
}

data_dir = args.data
# data_dir = '../material/hymenoptera_data'
folder_dict = {'train': 'train', 'stable_train': 'train', 'val': 'val'}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, folder_dict[x]),
                                          data_transforms[x])
                  for x in ['train', 'stable_train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True if x=='train' and args.arch != 'knn' else False, num_workers=args.workers)
              for x in ['train', 'stable_train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'stable_train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Build my own dataset
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def baseline():
    # print(args)
    # Only fine tune the last layer (alexnet.classifier[6])
    alexnet = models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_ftrs, len(class_names))

    alexnet = alexnet.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.SGD(alexnet.classifier[6].parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.step_size, gamma=0.1)
    
    model = train_model(alexnet, criterion, optimizer_conv,
                        exp_lr_scheduler, num_epochs=args.epochs)
    return model

def baseline_val(model_ckpt):
    since = time.time()
    
    alexnet = models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_ftrs, len(class_names))

    alexnet = alexnet.to(device)
    alexnet.load_state_dict(model_ckpt)
    alexnet.eval()

    running_corrects = 0
    result = []
    # Iterate over data. 
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = alexnet(inputs)
        _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        result.append(outputs.cpu().data.numpy())

    acc = running_corrects.double() / dataset_sizes['val']

    print('Checkpoint: {}, Acc: {:.4f}'.format(args.ckpt, acc))

    print('')
    return np.concatenate(result,axis=0)


def train_model_for_predict_param(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Building own dataset
    train_data_loader = ParamLearnerDataLoader(image_datasets['train'], len(class_names))
    if args.special_loader:
        print('use special loader')
        dataloaders['train'] = train_data_loader

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Calculate R
    Rs = []
    for i in range(len(class_names)):
        Rs.append([])
    model.eval()
    for inputs, labels in dataloaders['stable_train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        for index in range(len(inputs)):
            r = model.get_r(inputs[index].unsqueeze_(0))
            Rs[labels[index]].append(r.squeeze())
    for i in range(len(class_names)):
        rmean = torch.mean(torch.stack(Rs[i]), dim=0)
        Rs[i].append(rmean)
        Rs[i] = torch.stack(Rs[i])
        Rs[i].requires_grad = False
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data. 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    R = []
                    for i in range(len(class_names)):
                        index = np.random.randint(low=0, high=Rs[i].shape[0])
                        R.append(Rs[i][index])
                    R = torch.stack(R)
                    if phase == 'train':
                        outputs = model(inputs, R)
                    else:
                        outputs = model.forward_test(inputs, Rs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def predict_param():
    model = ParamLearner(drop_rate=args.drop_rate)
        
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.param_learner.parameters(), lr=args.lr, momentum=args.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    model = train_model_for_predict_param(model, criterion, optimizer,
                        exp_lr_scheduler, num_epochs=args.epochs)
    return model

def predict_param_val(model_ckpt):
    since = time.time()
    
    model = ParamLearner()
    model = model.to(device)
    model.load_state_dict(model_ckpt)
    model.eval()
    result = []

    # Calculate R
    Rs = []
    for i in range(len(class_names)):
        Rs.append([])
    for inputs, labels in dataloaders['stable_train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        for index in range(len(inputs)):
            r = model.get_r(inputs[index].unsqueeze_(0))
            Rs[labels[index]].append(r.squeeze())
    for i in range(len(class_names)):
        rmean = torch.mean(torch.stack(Rs[i]), dim=0)
        Rs[i].append(rmean)
        Rs[i] = torch.stack(Rs[i])
        print(Rs[i].shape)
    print(len(Rs))
    running_corrects = 0

    # Iterate over data. 
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model.forward_test(inputs, Rs)
        _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        result.append(outputs.cpu().data.numpy())
        
    acc = running_corrects.double() / dataset_sizes['val']

    print('Checkpoint: {}, Acc: {:.4f}'.format(args.ckpt, acc))

    print('')
    return np.concatenate(result,axis=0)

def KNN():
    model = models.alexnet(pretrained=True)
    # Delete last fc layer
    model.classifier.__delitem__(6)
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()
    since = time.time()
    feature = []
    label = []
    # Iterate over data.
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        ft = model(inputs)
        feature.append(ft)
        label.append(labels)
    feature = torch.cat(feature, dim=0)
    label = torch.cat(label, dim=0)
    print('feature shape: ', feature.shape)
    print('label shape:', label.shape)
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return {'feature':feature, 'label': label}

def find_knn_l2(X, selfX, selfY, k):
    num_test = X.shape[0]
    num_train = selfX.shape[0]
    dists = torch.sum(selfX**2, dim=1)\
          + torch.sum(X**2, dim=1).view((num_test,1))\
          - 2*X.mm(selfX.t())
    y_pred = torch.zeros(num_test, dtype=torch.long)
    for i in range(num_test):
        _, ind = torch.sort(dists[i])
        closest_y = selfY[ind[:k]]
        y_pred[i] = np.bincount(closest_y.cpu().data.numpy()).argmax()
    return y_pred

def find_knn_cos(X, selfX, selfY, k):
    num_test = X.shape[0]
    num_train = selfX.shape[0]
    y_pred = torch.zeros(num_test, dtype=torch.long)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(num_test):
        dists = cos(X[i:i+1], selfX)
        _, ind = torch.sort(dists)
        closest_y = selfY[ind[-k:]]
        y_pred[i] = np.bincount(closest_y.cpu().data.numpy()).argmax()
    return y_pred

def KNN_val(data):
    feature, label = data['feature'], data['label']
    feature = feature.to(device)
    label = label.to(device)
    since = time.time()
    model = models.alexnet(pretrained=True)
    # Delete last fc layer
    model.classifier.__delitem__(6)
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()
    result = []
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = find_knn_cos(outputs, feature, label, args.k)
        # statistics
        running_corrects += torch.sum(preds.to(device) == labels.data)
        result.append(outputs.cpu().data.numpy())

    epoch_acc = running_corrects.double() / dataset_sizes['val']
    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s Acc: {:.4f}'.format(
        time_elapsed // 60, time_elapsed % 60, epoch_acc))

    return np.concatenate(result,axis=0)


if __name__ == '__main__':
    if args.arch not in ['baseline', 'parampred', 'knn']:
        raise ValueError('args.arch %s not in [baseline, parampred, knn]'%args.arch)
    if args.prefix == '' and args.ckpt is None:
        raise ValueError('Please specify args.prefix!')
    if args.ckpt is None:
        # train
        if args.arch == 'baseline':
            model = baseline()
            torch.save(model.state_dict(), 'baseline_%s.pth' % args.prefix)
        elif args.arch == 'parampred':
            model = predict_param()
            torch.save(model.state_dict(), 'parampred_%s.pth' % args.prefix)
        else:
            feature = KNN()
            torch.save(feature, 'knn_%s.pth' % args.prefix)
    else:
        # val
        model = torch.load(args.ckpt)
        if args.arch == 'baseline':
            labels = baseline_val(model)
        elif args.arch == 'parampred':
            labels = predict_param_val(model)
        else:
            labels = KNN_val(model)
        with open('%s.txt'%(args.ckpt.replace('.pth','')),'w') as f:
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    f.write('%lf '%labels[i,j])
                f.write('\n')
# for i in $(seq 1 15);do CUDA_VISIBLE_DEVICES=7 ./main.py --step_size 30 --prefix 7_$i --lr 0.0001;done
