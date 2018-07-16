#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo
import random

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ParamLearner(models.AlexNet):
    def __init__(self, hidden_size=4096, drop_rate=0.5):
        super(ParamLearner, self).__init__() 
        self.drop_rate = drop_rate
        self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        for param in self.parameters():
            param.requires_grad = False

        # Delete last fc layer
        self.classifier.__delitem__(6)

        # Define param learner
        self.param_learner = nn.Linear(hidden_size, hidden_size)

        # Initialized with identity matrix
        self.param_learner.weight.data.copy_(torch.eye(hidden_size))  
    
    def forward(self, x, R):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        # R is a (num_class, hidden_size) matrix, w is a (num_class, hidden_size) matrix
        w = self.param_learner(R) 
        dr = torch.nn.Dropout(p=self.drop_rate)
        x = torch.matmul(x, w.transpose(0, 1))
        return x
    
    def get_r(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def forward_test(self, x, Rs):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        logits = [] 
        for i in range(x.shape[0]):
            i_logits = []
            for class_Rs in Rs:
                # class_Rs is a (n, hidden_size) matrix, in which n is the number of training pictures of this class.
                class_w = self.param_learner(class_Rs) 
                class_logit = torch.matmul(class_w, x[i])
                i_logits.append(class_logit.max())
            logits.append(torch.stack(i_logits))
        return torch.stack(logits)

class ParamLearnerDataLoader(object):
    def __init__(self, data_folder, num_classes):
        self.data_folder = data_folder
        self.data = []
        self.num_classes = num_classes
        for i in range(num_classes):
            self.data.append([])
        for i in range(data_folder.__len__()):
            img, label = data_folder.__getitem__(i)
            self.data[label].append(i)
        self.max_batch = 1e9
        for i in range(num_classes):
            self.max_batch = min(self.max_batch, len(self.data[i]))
        self.index = 0

    def __next__(self):
        if self.index >= self.max_batch:
            self.index = 0
            for i in range(self.num_classes):
                random.shuffle(self.data[i])
            raise StopIteration
        
        inputs = []
        labels = []

        for i in range(self.num_classes):
            image, label = self.data_folder.__getitem__(self.data[i][index])
            inputs.append(image)
            labels.append(label)

        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        index += 1

        return inputs, labels

    def next(self):
        if self.index >= self.max_batch:
            self.index = 0
            for i in range(self.num_classes):
                random.shuffle(self.data[i])
            raise StopIteration
        
        inputs = []
        labels = []

        for i in range(self.num_classes):
            image, label = self.data_folder.__getitem__(self.data[i][self.index])
            inputs.append(image)
            labels.append(label)

        inputs = torch.stack(inputs)
        labels = torch.tensor(labels, dtype=torch.long)
        self.index += 1

        return inputs, labels

        
    def __len__(self):
        return self.max_batch
    
    def __iter__(self):
        return self

