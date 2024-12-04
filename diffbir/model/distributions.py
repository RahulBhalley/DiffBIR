import torch
import numpy as np


class AbstractDistribution:
    """Abstract base class for probability distributions.
    
    This class defines the interface that all distribution classes should implement.
    """
    def sample(self):
        """Sample from the distribution.
        
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    def mode(self):
        """Get the mode (most likely value) of the distribution.
        
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    """A Dirac delta distribution that always returns a fixed value.
    
    The Dirac delta distribution is a degenerate distribution where the probability
    density function is zero everywhere except at a single point.

    Args:
        value: The fixed value that this distribution will always return.
    """
    def __init__(self, value):
        self.value = value

    def sample(self):
        """Sample from the Dirac distribution.
        
        Returns:
            The fixed value specified during initialization.
        """
        return self.value

    def mode(self):
        """Get the mode of the Dirac distribution.
        
        Returns:
            The fixed value specified during initialization.
        """
        return self.value


class DiagonalGaussianDistribution(object):
    """A diagonal Gaussian (normal) distribution with learnable parameters.
    
    This implements a multivariate normal distribution where the covariance matrix
    is diagonal, meaning dimensions are assumed to be independent.

    Args:
        parameters (torch.Tensor): Concatenated means and log-variances
        deterministic (bool, optional): If True, the distribution collapses to a 
            deterministic one at the mean. Defaults to False.
    """
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        """Sample from the distribution using the reparameterization trick.
        
        Returns:
            torch.Tensor: A sample from the distribution.
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        """Compute KL divergence between this distribution and another.
        
        Args:
            other (DiagonalGaussianDistribution, optional): Distribution to compute KL divergence with.
                If None, computes KL divergence with standard normal. Defaults to None.
                
        Returns:
            torch.Tensor: The KL divergence value.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        """Compute negative log likelihood of a sample.
        
        Args:
            sample (torch.Tensor): The sample to compute likelihood for
            dims (list, optional): Dimensions to sum over. Defaults to [1,2,3].
            
        Returns:
            torch.Tensor: The negative log likelihood value.
        """
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        """Get the mode of the distribution.
        
        For a Gaussian distribution, the mode equals the mean.
        
        Returns:
            torch.Tensor: The mode/mean of the distribution.
        """
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Compute the KL divergence between two diagonal Gaussian distributions.
    
    This function implements a numerically stable version of KL divergence computation
    between two diagonal Gaussian distributions.
    
    Args:
        mean1 (torch.Tensor): Mean of the first Gaussian
        logvar1 (torch.Tensor): Log variance of the first Gaussian
        mean2 (torch.Tensor): Mean of the second Gaussian
        logvar2 (torch.Tensor): Log variance of the second Gaussian
        
    Returns:
        torch.Tensor: The KL divergence value
        
    Note:
        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        
    Source:
        https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
