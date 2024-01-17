import torch

class Config:
    
    # seed
    seed = 42
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current vram: {device}')
    
    # repos
    data_dir = ''
    images_dir = '/Users/nagarajuoruganti/Downloads/Skin cancer ISIC The International Skin Imaging Collaboration'
    models_dir = ''
    model_path = ''
    
    # Folds
    n_splits = 5
    fold = -1
    target = 'category'
    target_labels = ['actinic keratosis',
                     'basal cell carcinoma',
                     'dermatofibroma',
                     'melanoma',
                     'nevus',
                     'pigmented benign keratosis',
                     'seborrheic keratosis',
                     'squamous cell carcinoma',
                     'vascular lesion']
    
    # Image preprocessing
    centered_zero = False
    in_channels = 3
    height = 180
    width = 180
    use_class_weights = False
    
    # Augmentation
    aug_threshold = 0
    max_rotation_angle = 30
    
    # Train parameters
    train_batch_size = 16
    valid_batch_size = 32
    iters_to_accumulate = 1
    learning_rate = 1e-4
    num_epochs = 1000
    
    # model params
    in_channels = 3
    
    # Run params
    sample_run = False
    save_epoch_wait = 1
    early_stop_count = 10
    save_checkpoint = True
    reload_checkpoint_on_lr_decrease = False
    
    # Misc
    show_folds_plot = False
    train_with_pre_resized_images = True
    
    # Data augmentation
    aug_dir = '/Users/nagarajuoruganti/Desktop/delete'
    aug_size = 0