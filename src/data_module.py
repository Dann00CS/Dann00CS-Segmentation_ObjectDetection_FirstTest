import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import os
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, num_classes, new_shape):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.new_shape = new_shape
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ix):
        resized_img = cv2.resize(self.X[ix], self.new_shape, interpolation=cv2.INTER_NEAREST)
        mask_oh = (np.arange(2) == self.y[ix][...,None]).astype(np.float32) 
        resized_mask_oh = cv2.resize(mask_oh, self.new_shape, interpolation=cv2.INTER_NEAREST)
        resized_img_tensor = torch.tensor(resized_img).permute(2,0,1)
        resized_mask_oh_tensor = torch.tensor(resized_mask_oh).permute(2,0,1)
        return resized_img_tensor, resized_mask_oh_tensor


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_classes, new_shape, dataset_csv_path, dataset_folder):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.new_shape = new_shape
        self.dataset_csv_path = dataset_csv_path
        self.dataset_folder = dataset_folder

    def setup(self, stage='Train'):
        # if stage=='Train':
        X, y = [], []
        train_df = pd.read_csv(self.dataset_csv_path)
        train_df = train_df[train_df['label']=='dog'] #Only dog
        for _, row in train_df.iterrows():
            img = Image.open(os.path.join(self.dataset_folder, 'images', row['image_path']))
            img = (np.array(img)/255).astype('float32')
            with open(os.path.join(self.dataset_folder, 'masks', row['mask_image_path']), 'rb') as f:
                img_mask = pickle.load(f).astype('int8')
            X.append(img), y.append(img_mask)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        print('shape: ', len(X_train), len(X_val))
        self.train_ds = Dataset(X_train, y_train, self.num_classes, self.new_shape)
        self.val_ds = Dataset(X_val, y_val, self.num_classes, self.new_shape)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


                

