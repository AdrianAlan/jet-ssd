import numpy as np
import torch


class EarlyStopping:
    '''Early stopping checkpont'''
    def __init__(self, patience=7, delta=10e-6,
                 save_best=True, save_path='./', verbose=False):
        self.patience = patience
        self.delta = delta
        self.save_best = save_best
        self.save_path = save_path
        self.verbose = verbose

        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False

        if self.verbose:
            print('Initiated early stopping with patience {}.'.format(
                  self.patience))

    def __call__(self, loss, model):
        '''Veryfy if training should be terminated'''

        if loss < self.best_score - self.delta:
            self.counter = 0  # Reset counter
            self.best_score = loss  # Store best score
            if self.save_best:
                self.save_checkpoint(model)
        else:
            self.counter += 1  # Increment counter
            if self.counter == self.patience:
                print('Stopped!')
                return True
        return False

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Checkpoint: saving model')
        torch.save(model.state_dict(), self.save_path)
