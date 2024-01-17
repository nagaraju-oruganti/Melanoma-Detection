import os
import random
import numpy as np
import pandas as pd
import torch
import glob
from PIL import Image
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('icefire')

from IPython.display import clear_output
import Augmentor

import shutil
    
### SEED EVERYTHING
def seed_everything(seed: int = 42):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print(f'set seed to {seed}')

### CREATE FOLDS
from sklearn.model_selection import StratifiedKFold 
def create_folds(config):
    
    data_path = os.path.join(config.data_dir, 'data.csv')
    
    # load train data
    data = pd.read_csv(data_path)
    data['kfold'] = -1
    
    # split into folds
    df = data[data['data_kind'] == 'Train']
    df.reset_index(inplace = True, drop = True)
    tdf = data[data['data_kind'] == 'Test']
    kf = StratifiedKFold(n_splits = config.n_splits,
                         random_state = config.seed,
                         shuffle = True)
    
    X = df.drop(columns = [config.target])
    y = df[config.target]
    
    # iterate over folds
    for i, (_, v_idx) in enumerate(kf.split(X = X, y = y), start = 1):
        df.loc[v_idx, ['kfold']] = i
        
    print('Samples in each fold:')
    print(df['kfold'].value_counts())
    
    # print('Distribution of target labels in each fold:')
    # labels = df[config.target].unique()
    # for f in range(1, config.n_splits + 1):
    #     dist = df[df['kfold'] == f][config.target].value_counts().to_dict()
    #     dist = {lbl : dist[lbl] for lbl in labels}
    #     print(f'Fold {f}: {dist}')
        
    # combine train and test dataset
    df = pd.concat([df, tdf], axis = 0)
    df.reset_index(inplace = True, drop = True)
    df.to_csv(data_path, index = False) # save
    
    return df

### MAKE AMENDMENTS TO THE CONFIG
def amend_config(config):
    config.raw_data_dir = os.path.join(config.data_dir, 'raw')
    config.processed_data_dir = os.path.join(config.data_dir, 'processed', str(config.seed))
    config.dest_path = os.path.join(config.models_dir, config.model_name)
    
    os.makedirs(config.processed_data_dir, exist_ok=True)
    os.makedirs(config.dest_path, exist_ok=True)
    if config.fold_dir != '': os.makedirs(config.fold_dir, exist_ok=True)
    if config.aug_dir != '': os.makedirs(config.aug_dir, exist_ok=True)
    
    return config

### RESIZE IMAGES AND SAVE
def resize_images(source_dir, dest_dir):
    data = []
    files = glob.glob(f'{source_dir}/*/*/*')
    for path in tqdm(files, total = len(files)):
        [_, sub_dir, category, file] = path.replace(source_dir, '').split('/')
        os.makedirs(os.path.join(dest_dir, sub_dir, category), exist_ok=True)
        image = Image.open(path)
        new_image = image.resize((180, 180))
        new_image.save(f'{dest_dir}/{sub_dir}/{category}/{file}')
        data.append((sub_dir, category, file))
    df = pd.DataFrame(data, columns = ['data_kind', 'category', 'filename'])
    return df
        
def class_weights_estimator(df):
    
    # https://naadispeaks.blog/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
    
    class_weights = []
    dist = dict(df['label'].value_counts())
    
    total_samples = len(df)
    for c in sorted(df['label'].unique()):
        class_weights.append(
            1 - (dist[c] / total_samples)
        )
        
    #print('class_weights: ', [round(c, 5) for c in class_weights])
    
    return class_weights

## Plot results
def plot_results(results):
    # epoch, train_loss, valid_loss, train_score, valid_score, lr
    df = pd.DataFrame(results, columns=['epoch', 'train_loss', 'valid_loss', 'train_score', 'valid_score', 'lr'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.lineplot(data = df, x = 'epoch', y = 'train_loss', color = 'blue', marker= 'o', ax = axes[0], label = 'Train loss')
    sns.lineplot(data = df, x = 'epoch', y = 'valid_loss', color = 'red', marker= 'o', ax = axes[0], label = 'Valid loss')
    axes[0].set(xlabel = 'Epoch', ylabel = 'Cross-entropy loss', title = 'Training and validation Loss')
    sns.lineplot(data = df, x = 'epoch', y = 'train_score', color = 'blue', marker= 'o', ax = axes[1], label = 'Train accuracy')
    sns.lineplot(data = df, x = 'epoch', y = 'valid_score', color = 'red', marker= 'o', ax = axes[1], label = 'Valid accuracy')
    axes[1].set(xlabel = 'Epoch', ylabel = 'Accuracy', title = 'Training and validation accuracy')
    for ax in axes:
        ax.legend(loc = 'upper right')
        ax.grid(linestyle = '--')
    plt.tight_layout()
    plt.show()

### Augmentation
def augment_data(config, LABEL_TO_INDEX_MAP):
    
    '''Note that the augmented data includes both original and augmented images'''
    
    if config.aug_size == 0:
        return []
    
    parent_path = os.path.join(config.fold_dir, 'Fold_Train')
    for label in config.target_labels:
        datapath = os.path.join(parent_path, label)
        output_dir = os.path.join(config.aug_dir, label)
        os.makedirs(output_dir, exist_ok = True)
        if len(os.listdir(output_dir)) != config.aug_size + len(os.listdir(datapath)):
            shutil.rmtree(output_dir)
            ## Augment data
            p = Augmentor.Pipeline(datapath, output_directory = output_dir)
            p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
            p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
            p.sample(config.aug_size)
            p.process()
            clear_output()          # clear output in Jupyter notebook

    ## Make augment dataset
    data = []
    print('================================================================')
    print('Images after augmentation:')
    for label in config.target_labels:
        label_dir = os.path.join(config.aug_dir, label)
        for path in os.listdir(label_dir):
            data_kind = 'Augmented'
            filename = path
            index = LABEL_TO_INDEX_MAP[label]
            filepath = f'{label_dir}/{filename}'
            data.append((filepath, filename, index, True))
            
        print(f'{label:40s} : {len(os.listdir(label_dir))}')
    print('================================================================')
            
    return data