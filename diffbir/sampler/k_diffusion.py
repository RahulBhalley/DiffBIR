"""
K-diffusion sampling algorithms and utilities.

This module implements various sampling algorithms and helper functions for K-diffusion,
following Karras et al. (2022).

Functions:
    append_dims: Append dimensions to a tensor until it reaches target dimensions.
    append_zero: Append a zero element to the end of a tensor.
    get_sigmas_karras: Construct noise schedule from Karras et al. (2022).
    get_sigmas_exponential: Construct exponential noise schedule.
    get_sigmas_polyexponential: Construct polynomial-exponential noise schedule.
    get_sigmas_vp: Construct continuous VP noise schedule.
    to_d: Convert denoiser output to Karras ODE derivative.
    get_ancestral_step: Calculate noise levels for ancestral sampling step.
    default_noise_sampler: Default noise sampling function.
    sample_euler: Euler method sampler.
    sample_euler_ancestral: Ancestral sampling with Euler steps.
    sample_heun: Heun method sampler.
"""

import math

from scipy import integrate
import torch
from torch import nn
import torchsde
from tqdm.auto import trange, tqdm


def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions.
    
    Args:
        x (torch.Tensor): Input tensor to append dimensions to
        target_dims (int): Desired number of dimensions
        
    Returns:
        torch.Tensor: Tensor with dimensions appended up to target_dims
        
    Raises:
        ValueError: If target_dims is less than input tensor's dimensions
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    """Append a zero element to the end of a tensor.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Tensor with zero appended
    """
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Construct the noise schedule of Karras et al. (2022).
    
    Args:
        n (int): Number of noise levels
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        rho (float, optional): Schedule parameter. Defaults to 7.0
        device (str, optional): Device to place tensors on. Defaults to 'cpu'
        
    Returns:
        torch.Tensor: Tensor of noise levels with shape (n+1,)
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Construct an exponential noise schedule.
    
    Args:
        n (int): Number of noise levels
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        device (str, optional): Device to place tensors on. Defaults to 'cpu'
        
    Returns:
        torch.Tensor: Tensor of noise levels with shape (n+1,)
    """
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Construct a polynomial in log sigma noise schedule.
    
    Args:
        n (int): Number of noise levels
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        rho (float, optional): Schedule parameter. Defaults to 1.0
        device (str, optional): Device to place tensors on. Defaults to 'cpu'
        
    Returns:
        torch.Tensor: Tensor of noise levels with shape (n+1,)
    """
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Construct a continuous VP noise schedule.
    
    Args:
        n (int): Number of noise levels
        beta_d (float, optional): Maximum noise level parameter. Defaults to 19.9
        beta_min (float, optional): Minimum noise level parameter. Defaults to 0.1
        eps_s (float, optional): Small constant. Defaults to 1e-3
        device (str, optional): Device to place tensors on. Defaults to 'cpu'
        
    Returns:
        torch.Tensor: Tensor of noise levels with shape (n+1,)
    """
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def to_d(x, sigma, denoised):
    """Convert a denoiser output to a Karras ODE derivative.
    
    Args:
        x (torch.Tensor): Input tensor
        sigma (torch.Tensor): Noise level
        denoised (torch.Tensor): Denoised output
        
    Returns:
        torch.Tensor: ODE derivative
    """
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculate the noise levels for an ancestral sampling step.
    
    Args:
        sigma_from (float): Starting noise level
        sigma_to (float): Target noise level
        eta (float, optional): Ancestral sampling parameter. Defaults to 1.0
        
    Returns:
        tuple: (sigma_down, sigma_up) - noise levels for stepping down and adding noise
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    """Create a default noise sampling function.
    
    Args:
        x (torch.Tensor): Template tensor for shape/device/dtype
        
    Returns:
        callable: Noise sampling function
    """
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy.
    
    Args:
        x (torch.Tensor): Template tensor for shape/device/dtype
        t0 (float): Start time
        t1 (float): End time
        seed (int or None, optional): Random seed. Defaults to None
        **kwargs: Additional arguments passed to BrownianTree
    """

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        """Sort two values and return sort direction."""
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        """Sample noise for the interval [t0, t1]."""
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (torch.Tensor): Template tensor for shape/device/dtype
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        seed (int or List[int], optional): Random seed(s). Defaults to None
        transform (callable, optional): Maps sigma to internal timestep. Defaults to identity
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        """Sample noise for transitioning from sigma to sigma_next."""
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022).
    
    Args:
        model (callable): Model function
        x (torch.Tensor): Input tensor
        sigmas (torch.Tensor): Noise schedule
        extra_args (dict, optional): Extra arguments for model. Defaults to None
        callback (callable, optional): Callback function. Defaults to None
        disable (bool, optional): Disable progress bar. Defaults to None
        s_churn (float, optional): Stochasticity parameter. Defaults to 0.0
        s_tmin (float, optional): Minimum sigma for stochasticity. Defaults to 0.0
        s_tmax (float, optional): Maximum sigma for stochasticity. Defaults to inf
        s_noise (float, optional): Noise scale. Defaults to 1.0
        
    Returns:
        torch.Tensor: Sampled tensor
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps.
    
    Args:
        model (callable): Model function
        x (torch.Tensor): Input tensor
        sigmas (torch.Tensor): Noise schedule
        extra_args (dict, optional): Extra arguments for model. Defaults to None
        callback (callable, optional): Callback function. Defaults to None
        disable (bool, optional): Disable progress bar. Defaults to None
        eta (float, optional): Ancestral sampling parameter. Defaults to 1.0
        s_noise (float, optional): Noise scale. Defaults to 1.0
        noise_sampler (callable, optional): Noise sampling function. Defaults to None
        
    Returns:
        torch.Tensor: Sampled tensor
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    
    Args:
        model (callable): Model function
        x (torch.Tensor): Input tensor
        sigmas (torch.Tensor): Noise schedule
        extra_args (dict, optional): Extra arguments for model. Defaults to None
        callback (callable, optional): Callback function. Defaults to None
        disable (bool, optional): Disable progress bar. Defaults to None
        s_churn (float, optional): Stochasticity parameter. Defaults to 0.0
        s_tmin (float, optional): Minimum sigma for stochasticity. Defaults to 0.0
        s_tmax (float, optional): Maximum sigma for stochasticity. Defaults to inf
        s_noise (float, optional): Noise scale. Defaults to 1.0
        
    Returns:
        torch.Tensor: Sampled tensor
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).

    This sampler combines ideas from DPM-Solver-2 and Karras et al.'s Algorithm 2 to provide
    high-quality sampling with second-order accuracy.

    Args:
        model (callable): Model function that takes (x, sigma) and returns predicted noise/score.
        x (torch.Tensor): Input tensor containing the initial noise.
        sigmas (torch.Tensor): Sequence of noise levels defining the diffusion schedule.
        extra_args (dict, optional): Extra arguments to pass to model. Defaults to None.
        callback (callable, optional): Function called after each step with diagnostic data. Defaults to None.
        disable (bool, optional): Disable progress bar. Defaults to None.
        s_churn (float, optional): Parameter controlling amount of stochasticity. Defaults to 0.0.
        s_tmin (float, optional): Minimum sigma where stochasticity is applied. Defaults to 0.0.
        s_tmax (float, optional): Maximum sigma where stochasticity is applied. Defaults to infinity.
        s_noise (float, optional): Scaling factor for noise. Defaults to 1.0.

    Returns:
        torch.Tensor: The denoised sample.

    Notes:
        The sampler uses Euler steps at sigma=0 and DPM-Solver-2 steps otherwise.
        Stochasticity is controlled by s_churn parameter and only applied when sigma is
        between s_tmin and s_tmax.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps.

    This sampler combines ancestral sampling with DPM-Solver-2 steps for improved sample quality.

    Args:
        model (callable): Model function that takes (x, sigma) and returns predicted noise/score.
        x (torch.Tensor): Input tensor containing the initial noise.
        sigmas (torch.Tensor): Sequence of noise levels defining the diffusion schedule.
        extra_args (dict, optional): Extra arguments to pass to model. Defaults to None.
        callback (callable, optional): Function called after each step with diagnostic data. Defaults to None.
        disable (bool, optional): Disable progress bar. Defaults to None.
        eta (float, optional): Parameter controlling the stochasticity. Defaults to 1.0.
        s_noise (float, optional): Scaling factor for noise. Defaults to 1.0.
        noise_sampler (callable, optional): Custom noise sampling function. Defaults to None.

    Returns:
        torch.Tensor: The denoised sample.

    Notes:
        - Uses DPM-Solver-2 steps for deterministic updates
        - Adds ancestral sampling noise after each step
        - Falls back to Euler method when sigma reaches 0
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def linear_multistep_coeff(order, t, i, j):
    """Compute linear multistep coefficients.

    Args:
        order (int): Order of the multistep method.
        t (ndarray): Time points.
        i (int): Current step index.
        j (int): Index for coefficient calculation.

    Returns:
        float: Coefficient for the linear multistep method.

    Raises:
        ValueError: If requested order is too high for the current step.

    Notes:
        Computes coefficients for linear multistep methods using Lagrange polynomial interpolation.
    """
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    """Linear multistep sampling.

    Implements sampling using linear multistep methods of arbitrary order.

    Args:
        model (callable): Model function that takes (x, sigma) and returns predicted noise/score.
        x (torch.Tensor): Input tensor containing the initial noise.
        sigmas (torch.Tensor): Sequence of noise levels defining the diffusion schedule.
        extra_args (dict, optional): Extra arguments to pass to model. Defaults to None.
        callback (callable, optional): Function called after each step with diagnostic data. Defaults to None.
        disable (bool, optional): Disable progress bar. Defaults to None.
        order (int, optional): Order of the linear multistep method. Defaults to 4.

    Returns:
        torch.Tensor: The denoised sample.

    Notes:
        - Uses linear multistep methods for high-order accuracy
        - Automatically adjusts order based on available steps
        - Computes coefficients using Lagrange polynomial interpolation
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


class PIDStepSizeController:
    """A PID controller for ODE adaptive step size control.

    Implements a proportional-integral-derivative controller for adaptive step size selection
    in numerical ODE integration.

    Args:
        h (float): Initial step size.
        pcoeff (float): Proportional gain coefficient.
        icoeff (float): Integral gain coefficient.
        dcoeff (float): Derivative gain coefficient.
        order (int, optional): Order of the integration method. Defaults to 1.
        accept_safety (float, optional): Safety factor for step acceptance. Defaults to 0.81.
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-8.

    Notes:
        - Uses PID control theory to adapt integration step sizes
        - Maintains history of error values for I and D terms
        - Includes safety factors and limiters for robustness
    """
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        """Apply limiting function to control factor.

        Args:
            x (float): Raw control factor.

        Returns:
            float: Limited control factor.
        """
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        """Propose new step size based on error estimate.

        Args:
            error (float): Error estimate from last step.

        Returns:
            bool: Whether to accept the step.
        """
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """DPM-Solver implementation following https://arxiv.org/abs/2206.00927.
    
    This class implements the DPM-Solver algorithm for sampling from diffusion models.
    It provides multiple solver variants including 1-step, 2-step, and 3-step methods,
    as well as adaptive step size control.

    Args:
        model (nn.Module): The score model to use for denoising
        extra_args (dict, optional): Extra arguments to pass to model. Defaults to None.
        eps_callback (callable, optional): Callback function called after each eps computation. Defaults to None.
        info_callback (callable, optional): Callback function for logging solver info. Defaults to None.

    Attributes:
        model (nn.Module): The underlying score model
        extra_args (dict): Extra arguments passed to model
        eps_callback (callable): Callback after eps computation  
        info_callback (callable): Callback for logging solver info

    Notes:
        - Implements both fixed step size and adaptive step size solvers
        - Supports forward and reverse sampling directions
        - Includes ancestral sampling variants with noise injection
        - Based on the paper "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"
    """

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        """Convert noise level to time.

        Args:
            sigma (torch.Tensor): Noise level

        Returns:
            torch.Tensor: Corresponding time value
        """
        return -sigma.log()

    def sigma(self, t):
        """Convert time to noise level.

        Args:
            t (torch.Tensor): Time value

        Returns:
            torch.Tensor: Corresponding noise level
        """
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        """Compute eps (noise prediction) with caching.

        Args:
            eps_cache (dict): Cache of previously computed eps values
            key (str): Key for caching the computed eps
            x (torch.Tensor): Input tensor
            t (torch.Tensor): Time value
            *args: Additional positional arguments for model
            **kwargs: Additional keyword arguments for model

        Returns:
            tuple: (eps tensor, updated cache dict)
        """
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        """Single step DPM-Solver.

        Args:
            x (torch.Tensor): Current state
            t (torch.Tensor): Current time
            t_next (torch.Tensor): Next time
            eps_cache (dict, optional): Cache of eps values. Defaults to None.

        Returns:
            tuple: (next state, updated eps cache)
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        """Two step DPM-Solver.

        Args:
            x (torch.Tensor): Current state
            t (torch.Tensor): Current time 
            t_next (torch.Tensor): Next time
            r1 (float, optional): Intermediate step location. Defaults to 1/2.
            eps_cache (dict, optional): Cache of eps values. Defaults to None.

        Returns:
            tuple: (next state, updated eps cache)
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        """Three step DPM-Solver.

        Args:
            x (torch.Tensor): Current state
            t (torch.Tensor): Current time
            t_next (torch.Tensor): Next time
            r1 (float, optional): First intermediate step location. Defaults to 1/3.
            r2 (float, optional): Second intermediate step location. Defaults to 2/3.
            eps_cache (dict, optional): Cache of eps values. Defaults to None.

        Returns:
            tuple: (next state, updated eps cache)
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=1., noise_sampler=None):
        """Fast DPM-Solver with fixed step size.

        Args:
            x (torch.Tensor): Initial state
            t_start (torch.Tensor): Start time
            t_end (torch.Tensor): End time
            nfe (int): Number of function evaluations
            eta (float, optional): Noise scale factor. Defaults to 0.
            s_noise (float, optional): Noise scale. Defaults to 1.
            noise_sampler (callable, optional): Custom noise sampler. Defaults to None.

        Returns:
            torch.Tensor: Final state

        Raises:
            ValueError: If eta != 0 for reverse sampling
        """
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            denoised = x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else:
                x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)

            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        """Adaptive step size DPM-Solver.

        Args:
            x (torch.Tensor): Initial state
            t_start (torch.Tensor): Start time
            t_end (torch.Tensor): End time
            order (int, optional): Solver order (2 or 3). Defaults to 3.
            rtol (float, optional): Relative tolerance. Defaults to 0.05.
            atol (float, optional): Absolute tolerance. Defaults to 0.0078.
            h_init (float, optional): Initial step size. Defaults to 0.05.
            pcoeff (float, optional): P coefficient for PID control. Defaults to 0.
            icoeff (float, optional): I coefficient for PID control. Defaults to 1.
            dcoeff (float, optional): D coefficient for PID control. Defaults to 0.
            accept_safety (float, optional): Safety factor for step acceptance. Defaults to 0.81.
            eta (float, optional): Noise scale factor. Defaults to 0.
            s_noise (float, optional): Noise scale. Defaults to 1.
            noise_sampler (callable, optional): Custom noise sampler. Defaults to None.

        Returns:
            tuple: (final state, info dict)

        Raises:
            ValueError: If order not in {2,3} or eta != 0 for reverse sampling
        """
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast with fixed step size.

    A fast sampling method using DPM-Solver with fixed step sizes. See https://arxiv.org/abs/2206.00927 for details.

    Args:
        model: The model to sample from.
        x (torch.Tensor): Initial noise tensor.
        sigma_min (float): Minimum sigma/noise level.
        sigma_max (float): Maximum sigma/noise level.
        n (int): Number of sampling steps.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.
        eta (float, optional): Noise scale factor. Must be 0 for reverse sampling. Defaults to 0.
        s_noise (float, optional): Amount of noise to add each step. Defaults to 1.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.

    Returns:
        torch.Tensor: The sampled output tensor.

    Raises:
        ValueError: If sigma_min or sigma_max is 0.
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 with adaptive step size.

    An adaptive step size sampling method using DPM-Solver orders 12 and 23. See https://arxiv.org/abs/2206.00927 for details.

    Args:
        model: The model to sample from.
        x (torch.Tensor): Initial noise tensor.
        sigma_min (float): Minimum sigma/noise level.
        sigma_max (float): Maximum sigma/noise level.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.
        order (int, optional): Order of solver (2 or 3). Defaults to 3.
        rtol (float, optional): Relative tolerance for step size control. Defaults to 0.05.
        atol (float, optional): Absolute tolerance for step size control. Defaults to 0.0078.
        h_init (float, optional): Initial step size. Defaults to 0.05.
        pcoeff (float, optional): Proportional coefficient for PID controller. Defaults to 0.
        icoeff (float, optional): Integral coefficient for PID controller. Defaults to 1.
        dcoeff (float, optional): Derivative coefficient for PID controller. Defaults to 0.
        accept_safety (float, optional): Safety factor for step acceptance. Defaults to 0.81.
        eta (float, optional): Noise scale factor. Must be 0 for reverse sampling. Defaults to 0.
        s_noise (float, optional): Amount of noise to add each step. Defaults to 1.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.
        return_info (bool, optional): Whether to return solver statistics. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, dict]]: The sampled output tensor, and optionally solver statistics.

    Raises:
        ValueError: If sigma_min or sigma_max is 0.
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps.

    A sampling method combining ancestral sampling with DPM-Solver++(2S) second-order steps.

    Args:
        model: The model to sample from.
        x (torch.Tensor): Initial noise tensor.
        sigmas (torch.Tensor): Sequence of noise levels.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.
        eta (float, optional): Noise scale factor. Defaults to 1.
        s_noise (float, optional): Amount of noise to add each step. Defaults to 1.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.

    Returns:
        torch.Tensor: The sampled output tensor.
    """
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic).

    A stochastic sampling method using DPM-Solver++.

    Args:
        model: The model to sample from.
        x (torch.Tensor): Initial noise tensor.
        sigmas (torch.Tensor): Sequence of noise levels.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.
        eta (float, optional): Noise scale factor. Defaults to 1.
        s_noise (float, optional): Amount of noise to add each step. Defaults to 1.
        noise_sampler (callable, optional): Function to sample noise. Defaults to None.
        r (float, optional): Step size multiplier. Defaults to 1/2.

    Returns:
        torch.Tensor: The sampled output tensor.
    """
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M).

    A sampling method using DPM-Solver++(2M).

    Args:
        model: The model to sample from.
        x (torch.Tensor): Initial noise tensor.
        sigmas (torch.Tensor): Sequence of noise levels.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.

    Returns:
        torch.Tensor: The sampled output tensor.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE sampling method.

    A sampling method using DPM-Solver++(2M) with stochastic differential equations (SDE).

    Args:
        model: A model that takes in noisy samples and noise levels and returns denoised samples.
        x (torch.Tensor): Initial noise tensor to start sampling from.
        sigmas (torch.Tensor): Sequence of noise levels defining the diffusion process.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.
        eta (float, optional): Parameter controlling the stochastic noise. Set to 0 for deterministic sampling. Defaults to 1.0.
        s_noise (float, optional): Scale factor for the noise. Defaults to 1.0.
        noise_sampler (callable, optional): Function to sample noise. If None, uses BrownianTreeNoiseSampler. Defaults to None.
        solver_type (str, optional): Type of solver to use - either 'heun' or 'midpoint'. Defaults to 'midpoint'.

    Returns:
        torch.Tensor: The sampled output tensor.

    Raises:
        ValueError: If solver_type is not 'heun' or 'midpoint'.
    """

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE sampling method.

    A sampling method using DPM-Solver++(3M) with stochastic differential equations (SDE). This is a higher-order variant
    of DPM-Solver++(2M) that uses three previous model evaluations for better accuracy.

    Args:
        model: A model that takes in noisy samples and noise levels and returns denoised samples.
        x (torch.Tensor): Initial noise tensor to start sampling from.
        sigmas (torch.Tensor): Sequence of noise levels defining the diffusion process.
        extra_args (dict, optional): Extra arguments to pass to the model. Defaults to None.
        callback (callable, optional): Function called after each step with sampling info. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to None.
        eta (float, optional): Parameter controlling the stochastic noise. Set to 0 for deterministic sampling. Defaults to 1.0.
        s_noise (float, optional): Scale factor for the noise. Defaults to 1.0.
        noise_sampler (callable, optional): Function to sample noise. If None, uses BrownianTreeNoiseSampler. Defaults to None.

    Returns:
        torch.Tensor: The sampled output tensor.
    """

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x
