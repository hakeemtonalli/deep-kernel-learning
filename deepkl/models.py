import torch
import torch.nn as nn
import gpytorch
import numpy as np

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
        self.covar_module.initialize_from_data_empspect(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
class GRUFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRUFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        return out
        
class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=False)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        return out
    
def create_sequences(x, sequence_length):
    """
    Creates a transformed dataset for the LSTM with a new sequence length.
    """
    x_seq, y_seq = [], []
    for i in range(len(x) -sequence_length):
        x_seq.append(x[i: i + sequence_length])
        y_seq.append(x[i+1: i + sequence_length + 1])
    x_seq = np.array(x_seq).reshape(-1, sequence_length)
    y_seq = np.array(y_seq).reshape(-1, sequence_length)
    return x_seq, y_seq
    
if __name__ == '__main__':
    # generate univariate data
    y = np.random.normal(0, 1, 120)
    sequence_length = 5
    
    test_x, test_y = create_sequences(y, sequence_length=5)
    print(test_x)
    print(test_y.shape)
    
    
    # convert to Tensor and reshape to (seq, batch, dim)
    y = torch.Tensor(torch.from_numpy(test_y).float())
    y = y.view([len(y), -1, 5])
    print(y.size())
    
    # test forward method
    model = LSTMFeatureExtractor(input_dim=5, hidden_dim=16, num_layers=2)
    yhat = model(y)
    print(yhat)

    
    