import torch
import torch.nn as nn
from torchsummary import summary
    
class BaselineConvNet(nn.Module):
    def __init__(self, config, device):
        super(BaselineConvNet, self).__init__()
        self.config = config
        self.class_weights =  torch.tensor(config.class_weights, dtype = torch.float32).to(device)
        
        ## Convolutions
        net = []
        in_channels = self.config.in_channels
        for out_channels, p in zip(self.config.model_arch_depth, self.config.dropout_map):
            sub = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size = 2, stride = 2)]
            if p >0:
                sub.append(nn.Dropout(p))
            net.extend(sub)
            in_channels = out_channels
        net.append(nn.Flatten())
        self.network = nn.Sequential(*net)
        
        ## estimate units out of convolutions
        self.in_fc_units = self.estimate_fc_units()

        ## Fully connected layers
        fully_connected = []
        if len(self.config.dropout_fc) > 0:
            fully_connected.append(nn.Dropout(self.config.dropout_fc[0]))
        fully_connected.extend([nn.Linear(self.in_fc_units, 128), 
                                nn.ReLU()])
        if len(self.config.dropout_fc) > 1:
            fully_connected.append(nn.Dropout(self.config.dropout_fc[-1]))
        fully_connected.append(nn.Linear(128, len(config.target_labels)))
        self.fcs = nn.Sequential(*fully_connected)
        
        self.print_model_struture()
        
    def estimate_fc_units(self):
        sample = torch.randn(self.config.train_batch_size,
                             self.config.in_channels,
                             self.config.width,
                             self.config.height)
        return self.network(sample).shape[-1]
    
    def print_model_struture(self):
        
        print('=' * 120)
        print('>>> MODEL ARCHITECTURE')
        print(self)
        num_params: int = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('=' * 120)
        print(f'Total number of train parameters: {num_params} or ({round(num_params / 1e6, 2)}M)')
        print('=' * 120)
        
    def loss_fn(self, logits, y):
        if self.config.use_class_weights:
            criterion = nn.CrossEntropyLoss(weight = self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        return loss
        
    def forward(self, x, y):
        convs = self.network(x)
        logits = self.fcs(convs)
        loss = self.loss_fn(logits= logits, y = y)
        return logits, loss
