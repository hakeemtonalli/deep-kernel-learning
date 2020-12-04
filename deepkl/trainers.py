from deepkl.models import SpectralMixtureGPModel
import torch
import gpytorch


class SpectralMixtureTrainer:
    """
    Train Gaussian process with Spectral Mixture kernel 
    
    params:
    -------
        model (gpytorch.models): Gaussian process model
        loss (gpytorch.mlls): Likelihood function to optimize
        train_loader (torch DataLoader): Dataloader with training dataset for batch training. If None, assumed using train_batch() only.
        val_loader(torch DataLoader): Dataloader with validation dataset for batch training. 
        
    attributes:
    -----------
        model: see params
        likelihood (ExactGP.likelihood): Likelihood of the model
        to optimize.
    
    """
    
    def __init__(self, model, likelihood, train_loader=None, val_loader=None, optimizer=None, lr=0.1, device='cpu'):
        self.model = model
        self.likelihood = likelihood
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.device = device
        if val_loader is not None:
            self.val_iter = iter(val_loader)
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
    
        
    def train_batch(self, train_x, train_y, epochs=50):
        """ 
        Train model over a batch of observations 
        
        """
        self.model.train()
        self.likelihood.train()
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print(f"Epoch {i+1}/{epochs} - Loss: {loss.item()}")
            self.optimizer.step()
            
        self.model.eval()
        self.likelihood.eval()
        
        