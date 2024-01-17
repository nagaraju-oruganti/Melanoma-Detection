## Standard libraries
import os
import pandas as pd
import numpy as np
import random
import glob

## Pytorch and Image libraries
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

## Local modules
import utils
import shutil

from tqdm import tqdm

## Preprocessing dataset
class Preprocess:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.seed = config.seed
        self.fold = config.fold
        
    # Split original train dataset on fold into train and valid
    def split_data(self):
        
        if self.fold != -1:
            # load fold if exists othewise create folds
            if 'kfold' not in self.df.columns: 
                self.df = utils.create_folds(self.config)
                
            # plot folds
            if self.config.show_folds_plot: 
                self.plot_folds()
            
        # Map class labels
        self.INDEX_TO_LABEL_MAP = {idx:label for idx, label in enumerate(self.config.target_labels)}
        self.LABEL_TO_INDEX_MAP = {label:idx for idx, label in self.INDEX_TO_LABEL_MAP.items()}
        
        self.df['label'] = self.df[self.config.target].map(self.LABEL_TO_INDEX_MAP)
        
        self.class_weights = utils.class_weights_estimator(self.df)
        
    # plot folds
    def plot_folds(self):
        plt.figure(figsize = (12, 4))
        d = self.df.groupby(by = ['data_kind', 'category', 'kfold'])['filename'].count().reset_index()
        sns.barplot(data = d, x = 'category', y = 'filename', hue = 'data_kind', palette = 'rainbow')
        plt.xticks(rotation = 90)
        plt.ylabel('# of images')
        plt.title('Distribution of samples between the Train and Test datasets')
        plt.show()
        
    def partition_images(self):
        source_dir = os.path.join(self.config.data_dir, 'resized_images', 'Train')
        df = self.df[self.df['data_kind'] == 'Train']
        que = tqdm(df.iterrows(), total = len(df), desc = f'Organizing images for fold {self.config.fold}')
        for _, row in que:
            kind = row['data_kind']
            category = row['category']
            filename = row['filename']
            fold = row['kfold']
            dest_dir = os.path.join(self.config.fold_dir, 
                                    'Fold_Valid' if fold == self.config.fold else 'Fold_Train',
                                    category)
            os.makedirs(dest_dir, exist_ok=True)
            if not os.path.exists(f'{dest_dir}/{filename}'):
                shutil.copy2(f'{source_dir}/{category}/{filename}', f'{dest_dir}/{filename}')
        
    def make_datasets(self):
        
        # resize original images
        dest_dir = os.path.join(self.config.data_dir, 'resized_images')
        os.makedirs(dest_dir, exist_ok=True)
        if not os.path.exists(f'{self.config.data_dir}/data.csv'):
            print('Resizing original images to 180 x 180')
            self.df = utils.resize_images(source_dir = self.config.images_dir,
                                          dest_dir = dest_dir)
            self.df.to_csv(f'{self.config.data_dir}/data.csv', index = False)
        else:
            print('loading images from resized folder')
            self.df = pd.read_csv(f'{self.config.data_dir}/data.csv')
        
        # split data
        self.split_data()
        
        # Make fold dataset
        self.partition_images()
        
        # make datasets
        data = {'Train': [], 'Valid': []}
        for kind in data:
            if kind == 'Train': df = self.df[~self.df['kfold'].isin([self.config.fold, -1])]
            if kind == 'Valid': df = self.df[self.df['kfold'] == self.config.fold]
            for _, row in df.iterrows():
                data_kind = row['data_kind']
                category = row['category']
                filename = row['filename']
                label = row['label']
                filepath = f'{self.config.fold_dir}/Fold_{kind}/{category}/{filename}'
                data[kind].append((filepath, filename, label, False)) # target_name, image_filename, label, is_augmented
        
        # augmented train data
        data['augmented_train'] = utils.augment_data(config = self.config, LABEL_TO_INDEX_MAP = self.LABEL_TO_INDEX_MAP)
                
        self.data = data
        
### CROP DATASET
class CropDataset(Dataset):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.centered_zero = config.centered_zero
        self.height = config.height
        self.width = config.width
        self.max_rotation_angle = config.max_rotation_angle
        
    def __len__(self):
        return len(self.data)
    
    def read_image(self, path):
        # load resized image
        if self.config.train_with_pre_resized_images:
            return Image.open(path)
        return transforms.functional.resize(Image.open(path), size = [self.height, self.width])
    
    def __getitem__(self, idx):
        filepath, filename, label, shall_augment = self.data[idx]
        img = self.read_image(path = filepath)
        img_tensor = self.apply_transformation(img = img, shall_augment = shall_augment) / 255.
        if self.centered_zero:
            img_tensor = (img_tensor - 0.5) * 2
        label_tensor = torch.tensor(label, dtype = torch.long)
        return filename, img_tensor, label_tensor
        
    def apply_transformation(self, img, shall_augment):
        composer = transforms.Compose([
            transforms.PILToTensor(),      # automatically convert into [0, 1]
        ])
        return composer(img)
        
### DATALOADERS
def dataloaders(config):
    prep = Preprocess(config = config)
    prep.make_datasets()
    
    train_dataset = prep.data['Train'] if config.aug_size == 0 else prep.data['augmented_train']
    valid_dataset = prep.data['Valid']
    class_weights = prep.class_weights
    
    if config.sample_run:
        train_dataset = train_dataset[:24]
        valid_dataset = valid_dataset[:24]
       
    train = CropDataset(config = config, data = train_dataset)
    valid = CropDataset(config = config, data = valid_dataset)
    
    train_loader = DataLoader(train, 
                              batch_size = config.train_batch_size,
                              shuffle = True,
                              drop_last = False)
    valid_loader = DataLoader(valid,
                              batch_size = config.valid_batch_size,
                              shuffle = False,
                              drop_last = False)
    
    print(f'Samples in train dataset: {len(train_loader.dataset)}')
    print(f'Samples in test dataset: {len(valid_loader.dataset)}')
    return train_loader, valid_loader, class_weights