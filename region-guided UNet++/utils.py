from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import copy

'''
The code to preprocessing and load dataset
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(root):
    imgs=[]
    for filename in os.listdir(root):
        form = filename.split('_')[0]
        if form == 'image':
            tag = filename.split('_')      
            img = os.path.join(root, filename)
            mask1 = os.path.join(root,'binary_label1_' + tag[1])
            mask2 = os.path.join(root,'binary_label2_' + tag[1])
            imgs.append((img,mask1,mask2))
    return imgs

class Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y1_path, y2_path = self.imgs[index]

        img_x = Image.open(x_path)
        img_y1 = Image.open(y1_path)
        img_y2 = Image.open(y2_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y1 = img_y1.convert('L')
            img_y2 = img_y2.convert('L')
            img_y1 = self.target_transform(img_y1)
            img_y2 = self.target_transform(img_y2)
        return img_x, img_y1, img_y2

    def __len__(self):
        return len(self.imgs)

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.ToTensor()
])

def train_model(model, criterion, optimizer, dataloader_train, dataloader_val, num_epochs=20, patience=30): # use early stop
    min_val_loss = float('inf')
    best_epoch = 0
    best_model = None
    for epoch in range(num_epochs):
        dt_size = len(dataloader_train.dataset)
        # ----------------------TRAIN-----------------------
        model.train()
        epoch_loss = 0
        step = 0
        for x, y, c in dataloader_train:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            labels2 = c.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = 0.5 * criterion(outputs[0], labels) + 0.5 * criterion(outputs[1], labels2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataloader_train.batch_size + 1, loss.item()))
        print("epoch %d training loss:%0.3f" % (epoch, epoch_loss/step))
        # ----------------------VALIDATION-----------------------
        with torch.no_grad():
            model.eval()
            epoch_loss = 0
            step = 0
            for x, y, c in dataloader_val:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                labels2 = c.to(device)
                outputs = model(inputs)
                loss = 0.7 * criterion(outputs[0], labels) + 0.3 * criterion(outputs[1], labels2)
                epoch_loss += loss.item()
            val_loss = epoch_loss/step
            print("epoch %d validation loss:%0.5f" % (epoch, val_loss))
        if val_loss < min_val_loss:
            best_epoch = epoch
            min_val_loss = val_loss
            #torch.save(model.state_dict(), './models/weights-epoch%d-val_loss%s.pth' % (epoch, val_loss))
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > patience:
            break
    print('Best validation loss%0.5f at epoch%s'% (min_val_loss, best_epoch))
    return best_model

def meanIOU_per_image(y_pred, y_true):
    '''
    Calculate the IOU, averaged across images
    '''
    y_pred = y_pred.astype('bool')
    y_true = y_true.astype('bool')
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    
    return np.sum(intersection) / np.sum(union)

class Score():
    def __init__(self, y_pred, y_true, size = 512, threshold = 0.5):
        self.TN = 0
        self.FN = 0
        self.FP = 0
        self.TP = 0
        self.y_pred = y_pred > threshold
        self.y_true = y_true
        self.threshold = threshold
        
        for i in range(0, size):
            for j in range(0, size):
                if self.y_pred[i,j] == 1:
                    if self.y_pred[i,j] == self.y_true[i][j]:
                        self.TP = self.TP + 1
                    else:
                        self.FP = self.FP + 1
                else:
                    if self.y_pred[i,j] == self.y_true[i][j]:
                        self.TN = self.TN + 1
                    else:
                        self.FN = self.FN + 1        
 
    def get_Se(self):
        return (self.TP)/(self.TP + self.FN)
    
    def get_Sp(self):
        return (self.TN)/(self.TN + self.FP)
    
    def get_Pr(self):
        return (self.TP)/(self.TP + self.FP)
    
    def F1(self):
        Pr = self.get_Pr()
        Se = self.get_Se()
        return (2*Pr*Se)/(Pr + Se)
    
    def G(self):
        Sp = self.get_Sp()
        Se = self.get_Se()
        return math.sqrt(Se*Sp)
    
    def IoU(self):
        Pr = self.get_Pr()
        Se = self.get_Se()
        return (Pr*Se) /(Pr + Se - Pr*Se)
    
    def DSC(self):
        return (2* self.TP)/(2* self.TP + self.FP + self.FN) 