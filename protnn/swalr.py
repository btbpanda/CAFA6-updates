import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR


class SWALRWrapper:
    def __init__(self, model, optimizer, swa_start=5, swa_lr=5e-4,
                 anneal_epochs=15, swa_mode='constant', cycle_epochs=1):
        """
        Wrapper for SWALR with flexible learning rate schedule

        Args:
            model: PyTorch model
            optimizer: optimizer instance
            swa_start: epoch to start SWA (default: 5)
            swa_lr: SWA learning rate (default: 5e-4)
            anneal_epochs: number of epochs for cosine annealing before SWA
            swa_mode: 'constant', 'cyclic', or 'cosine'
                - constant: fixed LR during SWA (simple, fast)
                - cyclic: cyclic LR during SWA (better exploration, standard practice)
                - cosine: continue cosine annealing during SWA
            cycle_epochs: epochs per cycle for cyclic mode
        """
        self.swa_model = AveragedModel(model)
        self.optimizer = optimizer
        self.swa_start = swa_start
        self.swa_mode = swa_mode
        self.current_epoch = 0
        self.train_loader = None

        # Initial learning rate from optimizer
        self.base_lr = optimizer.param_groups[0]['lr']

        # Pre-SWA scheduler: Cosine Annealing
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=anneal_epochs,
            eta_min=swa_lr if swa_mode == 'constant' else swa_lr * 0.5
        )

        # SWA scheduler based on mode
        if swa_mode == 'constant':
            # SWALR with constant learning rate
            self.swa_scheduler = SWALR(
                optimizer,
                swa_lr=swa_lr,
                anneal_epochs=1,
                anneal_strategy='cos'
            )
        elif swa_mode == 'cyclic':
            # Cyclic learning rate for better exploration
            self.swa_scheduler = CyclicLR(
                optimizer,
                base_lr=swa_lr * 0.5,  # min LR
                max_lr=swa_lr * 2.0,  # max LR
                step_size_up=cycle_epochs,
                mode='triangular',
                cycle_momentum=False
            )
        elif swa_mode == 'cosine':
            # Continue cosine annealing during SWA
            remaining_epochs = 20 - swa_start  # assuming 20 total epochs
            self.swa_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=swa_lr * 0.1
            )
        else:
            raise ValueError(f"Unknown swa_mode: {swa_mode}")

    def step(self, model):
        """Update SWA model and scheduler"""
        if self.current_epoch >= self.swa_start:
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.current_epoch += 1

    def get_final_model(self, original_model):
        """Get the final model (SWA if used, otherwise original)"""
        if self.current_epoch > self.swa_start:
            # Update batch norm statistics
            if self.train_loader is not None:
                print("Updating batch norm statistics for SWA model...")
                torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model)
            return self.swa_model
        return original_model

    def set_train_loader(self, train_loader):
        """Set train loader for BN update"""
        self.train_loader = train_loader

    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']