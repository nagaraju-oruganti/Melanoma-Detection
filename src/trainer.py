import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp              # mixed precision
from torch import autocast
from torchsummary import summary
from datetime import datetime

from sklearn.metrics import f1_score, accuracy_score

## Local imports
from models import BaselineConvNet
from dataset import dataloaders
import utils

import warnings
warnings.filterwarnings('ignore')

#### Evaluate
def evaluate(model, dataloader, device = 'cpu'):
    results = []
    y_trues, y_preds = [], []
    batch_loss_list = []
    model.eval()
    with torch.no_grad():
        for (filenames, inputs, targets) in dataloader:
            logits, loss = model(inputs.to(device), targets.to(device))
            batch_loss_list.append(loss.item())
            probs = F.softmax(logits, dim = 1)
            preds = torch.argmax(probs, dim = 1)
            
            preds = preds.to('cpu').numpy().tolist()
            probs = probs.to('cpu').numpy().tolist()
            targets = targets.to('cpu').numpy().tolist()

            # save
            for i, _ in enumerate(targets):
                
                item = [filenames[i], targets[i]]
                item.extend(probs[i])
                item.append(preds[i])
                
                results.append(item)
            
            y_trues.extend(targets)
            y_preds.extend(preds)
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    # scoring
    score = accuracy_score(y_true = y_trues, y_pred = y_preds)#, average = 'macro')
    eval_loss = np.mean(batch_loss_list)
    return score, eval_loss, results

#### Trainer
def trainer(config, model, train_loader, valid_loader, optimizer, scheduler):
    
    def update_que():
        que.set_postfix({
            'batch_loss'        : f'{loss.item():4f}',
            'epoch_loss'        : f'{np.mean(batch_loss_list):4f}',
            'learning_rate'     : optimizer.param_groups[0]["lr"],
            })
    
    def save_checkpoint(model, epoch, eval_results, best = False):
        if best:
            save_path = os.path.join(config.dest_path, f'model_{config.fold}.pth' if config.fold != -1 else 'model.pth')
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }
            torch.save(checkpoint, save_path)
            with open(os.path.join(config.dest_path, f'valid_results_{config.fold}.pkl' if config.fold != -1 else 'valid_results.pkl'), 'wb') as f:
                pickle.dump(eval_results, f)
            print(f'>>> [{datetime.now()}] - Checkpoint and predictions saved')
        
    def dis(x): return f'{x:.6f}'
        
    def run_evaluation_sequence(ref_score, counter):
        
        def print_result():
            print('')
            text =  f'>>> [{datetime.now()} | {epoch + 1}/{NUM_EPOCHS} | Early stopping counter {counter}] \n'
            text += f'    loss          - train: {dis(train_loss)}      valid: {dis(valid_loss)} \n'
            text += f'    accuracy      - train: {dis(train_score)}      valid: {dis(valid_score)} \n'
            text += f'    learning rate        : {optimizer.param_groups[0]["lr"]:.5e}'
            print(text + '\n')
        
        # Evaluation
        train_score, train_loss, _              = evaluate(model, train_loader, device) 
        valid_score, valid_loss, eval_results   = evaluate(model, valid_loader, device)
        
        # append results
        lr =  optimizer.param_groups[0]["lr"]
        results.append((epoch, train_loss, valid_loss, train_score, valid_score, lr))
        
        # Learning rate scheduler
        scheduler.step(valid_score)
        
        ### Save checkpoint
        if ((epoch + 1) > config.save_epoch_wait) and (config.save_checkpoint):
            save_checkpoint(model, epoch, eval_results, best = valid_loss < ref_score)
        
        # Tracking early stop
        counter = 0 if valid_score >= ref_score else counter + 1
        ref_score = max(ref_score, valid_score)
        done = counter >= config.early_stop_count
        
        # show results
        print_result()
        
        # Save results
        with open(os.path.join(config.dest_path, f'results{config.fold}.pkl' if config.fold != -1 else 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        return ref_score, counter, done 
    
    ### MIXED PRECISION
    scaler = amp.GradScaler()
    
    results = []
    device = config.device
    precision = torch.bfloat16 if str(device) == 'cpu' else torch.float16
    NUM_EPOCHS = config.num_epochs
    iters_to_accumulate = config.iters_to_accumulate
    
    # dummy value for placeholders
    ref_score, counter = 1e-3, 0
    train_loss, valid_loss, train_f1, valid_f1 = 0, 0, 0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()                       # put model in train mode
        batch_loss_list = []

        que = tqdm(enumerate(train_loader), total = len(train_loader), ncols=160, desc = f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        for i, (_, images, targets) in que:
            
            ###### TRAINING SECQUENCE            
            with autocast(device_type = str(device), dtype = precision):
                _, loss = model(images.to(device), targets.to(device))            # Forward pass
                loss = loss / iters_to_accumulate
            
            # - Accmulates scaled gradients    
            scaler.scale(loss).backward()                                        # backward pass (scaled loss)
            
            if (i + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)                                           # step
                scaler.update()
                optimizer.zero_grad()                                            # zero grad
            #######
            
            batch_loss_list.append(loss.item())
            
            # Update que status
            update_que()
        
        ### Run evaluation sequence
        ref_score, counter, done = run_evaluation_sequence(ref_score, counter)
        if done:
            return results
            
    return results

def save_config(config, path):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

from utils import amend_config
def train(config):
    device = config.device
    utils.seed_everything(seed = config.seed)
    config = amend_config(config= config)
    
    # dataloaders
    train_loader, valid_loader, class_weights = dataloaders(config = config)
    config.class_weights = class_weights
    print(f'Class weights: {class_weights}')
    
    # define model
    model = BaselineConvNet(config = config, device = device)
    model.to(device)
    
    # optmizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)#, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor=0.1, patience=5)
    
    # Trainer
    results = trainer(config, model, train_loader, valid_loader, optimizer, scheduler)
    
    ### SAVE RESULTS
    with open(os.path.join(config.dest_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
        
    ### Plot results
    utils.plot_results(results)