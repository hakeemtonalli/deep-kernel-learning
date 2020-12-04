import torch
import gpytorch

class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    """
    Gaussian process model with Spectral Mixture Kernel from Wilson et al. (2013).
    
    Used for univariate data. In our case, we (train_x, train_y) is a univariate
    time series.
    
    params:
    ------
        train_x (torch.Tensor): Tensor with the index of the time series
        train_y (torch.Tensor): Tensor with the values of the time series
        num_mixtures (int): Number of mixtures for the kernel.
        likelihood (gpytorch.likelihoods): Likelihood for the problem. For a 
        real-valued time series, we use GaussianLikelihood(). 
    """
    def __init__(self, train_x, train_y, num_mixtures, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    

        