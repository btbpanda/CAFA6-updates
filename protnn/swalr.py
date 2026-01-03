import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

class SWALRWrapper:
    def __init__(self, model, optimizer, swa_start=5, swa_lr=0.001, anneal_epochs=15, anneal_strategy='cos'):
        """
        Wrapper for SWALR with cosine annealing
        
        Args:
            model: PyTorch model
            optimizer: optimizer instance
            swa_start: epoch to start SWA (default: 5)
            swa_lr: SWA learning rate (default: 0.001) - MUCH LOWER!
            anneal_epochs: number of epochs for cosine annealing before SWA
            anneal_strategy: annealing strategy ('cos' or 'linear')
        """
        self.swa_model = AveragedModel(model)
        self.optimizer = optimizer
        self.swa_start = swa_start
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=1, anneal_strategy=anneal_strategy)
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=anneal_epochs, eta_min=0)
        self.current_epoch = 0
        
    def step(self, model):
        """Update SWA model and scheduler"""
        if self.current_epoch >= self.swa_start:
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.current_epoch += 1
    
    def get_final_model(self, model):
        """Get the final averaged model"""
        if self.current_epoch > self.swa_start:
            # Update batch norm statistics
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model)
            return self.swa_model
        return model
    
    def set_train_loader(self, train_loader):
        """Set train loader for BN update"""
        self.train_loader = train_loader
