# https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_pytorch.py
import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    """A wrapper class for the forward SDE (VP type) noise schedule.

    This class provides functionality for both discrete-time and continuous-time diffusion models.
    It handles the noise schedule and provides methods to compute various coefficients needed for
    the diffusion process.

    For discrete-time models, it implements piecewise linear interpolation for log_alpha_t.
    For continuous-time models, it implements the linear VPSDE noise schedule.

    The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N(alpha_t * x_0, sigma_t^2 * I).
    
    Attributes:
        schedule (str): The noise schedule type - either 'discrete' or 'linear'
        T (float): The ending time of the forward process, default is 1.0
        total_N (int): Total number of steps in the schedule
        log_alpha_array (torch.Tensor): For discrete schedule, stores log(alpha) values
        t_array (torch.Tensor): For discrete schedule, stores timesteps
        beta_0 (float): For linear schedule, the minimum beta value
        beta_1 (float): For linear schedule, the maximum beta value

    Args:
        schedule (str): The noise schedule type. Must be either 'discrete' or 'linear'.
        betas (torch.Tensor, optional): The beta array for discrete-time DPM.
        alphas_cumprod (torch.Tensor, optional): The cumulative product alphas for discrete-time DPM.
        continuous_beta_0 (float, optional): The smallest beta for linear schedule. Defaults to 0.1.
        continuous_beta_1 (float, optional): The largest beta for linear schedule. Defaults to 20.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.

    Note:
        For discrete-time DPMs, either betas or alphas_cumprod must be provided.
        For alphas_cumprod, note that it represents \hat{alpha_n} where:
            q_{t_n | 0}(x_{t_n} | x_0) = N(\sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I)
        Therefore alpha_{t_n} = \sqrt{\hat{alpha_n}} and log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n})

    Examples:
        >>> # For discrete-time DPMs with betas:
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        
        >>> # For discrete-time DPMs with alphas_cumprod:
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        
        >>> # For continuous-time DPMs with linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
    """

    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):
        if schedule not in ['discrete', 'linear']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).to(dtype=dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """Clips log-SNR values for numerical stability.

        For some beta schedules like cosine schedule, the log-SNR can have numerical issues.
        This method clips the log-SNR near t=T to ensure stability.

        Args:
            log_alphas (torch.Tensor): The log alpha values to clip
            clipped_lambda (float, optional): The minimum lambda value. Defaults to -5.1

        Returns:
            torch.Tensor: The clipped log alpha values
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """Computes log(alpha_t) for given timesteps.

        Args:
            t (torch.Tensor): Continuous-time labels in [0, T]

        Returns:
            torch.Tensor: log(alpha_t) values
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """Computes alpha_t for given timesteps.

        Args:
            t (torch.Tensor): Continuous-time labels in [0, T]

        Returns:
            torch.Tensor: alpha_t values
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """Computes sigma_t for given timesteps.

        Args:
            t (torch.Tensor): Continuous-time labels in [0, T]

        Returns:
            torch.Tensor: sigma_t values
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """Computes lambda_t = log(alpha_t) - log(sigma_t) for given timesteps.

        Args:
            t (torch.Tensor): Continuous-time labels in [0, T]

        Returns:
            torch.Tensor: lambda_t values
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """Computes timestep t for a given lambda_t value.

        Args:
            lamb (torch.Tensor): The lambda_t values

        Returns:
            torch.Tensor: Corresponding timesteps t in [0, T]
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
    cfg_rescale=False
):
    """Create a wrapper function for the noise prediction model.

    This wrapper adapts various types of diffusion models to work with DPM-Solver's continuous-time ODE solver.
    It handles different model parameterizations and guidance methods.

    Args:
        model (callable): The diffusion model to wrap. Expected signature varies by guidance_type:
            - For "uncond": model(x, t_input, **model_kwargs) -> noise|x_start|v|score
            - For "classifier": Same as "uncond"
            - For "classifier-free": model(x, t_input, cond, **model_kwargs) -> noise|x_start|v|score
        
        noise_schedule (NoiseScheduleVP): Object defining the noise schedule parameters.
        
        model_type (str, optional): The model's parameterization type. Defaults to "noise".
            - "noise": Model predicts added noise
            - "x_start": Model predicts clean data x_0
            - "v": Model predicts velocity (see Salimans & Ho, 2022)
            - "score": Model predicts score function
        
        model_kwargs (dict, optional): Additional arguments to pass to model. Defaults to {}.
        
        guidance_type (str, optional): Type of guidance to use. Defaults to "uncond".
            - "uncond": No guidance
            - "classifier": Classifier guidance (Dhariwal & Nichol, 2021)
            - "classifier-free": Classifier-free guidance (Ho & Salimans, 2022)
        
        condition (torch.Tensor, optional): Condition for guided sampling. Required for
            "classifier" and "classifier-free" guidance. Defaults to None.
        
        unconditional_condition (torch.Tensor, optional): Condition for unconditional path
            in classifier-free guidance. Defaults to None.
        
        guidance_scale (float, optional): Scale factor for guidance. Defaults to 1.0.
        
        classifier_fn (callable, optional): Classifier for guidance. Required for "classifier" guidance.
            Expected signature: classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits
        
        classifier_kwargs (dict, optional): Additional arguments for classifier_fn. Defaults to {}.
        
        cfg_rescale (bool, optional): Whether to use dynamic rescaling of guidance scale based on timestep.
            Only applies to classifier-free guidance. Defaults to False.

    Returns:
        callable: A wrapped model function with signature model_fn(x, t_continuous) -> noise_pred,
            suitable for use with DPM-Solver.

    References:
        - Salimans, T., & Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion Models.
          arXiv:2202.00512.
        - Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis.
          Advances in Neural Information Processing Systems, 34.
        - Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance.
          arXiv:2207.12598.
    """

    def get_model_input_time(t_continuous):
        """Convert continuous time to model's expected time format.

        Args:
            t_continuous (torch.Tensor): Time in [epsilon, T] range

        Returns:
            torch.Tensor: Time in model's expected format:
                - For discrete models: Maps [1/N, 1] to [0, 1000*(N-1)/N]
                - For continuous models: Returns t_continuous unchanged
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        """Get noise prediction from model and convert to noise if needed.

        Args:
            x (torch.Tensor): Input tensor
            t_continuous (torch.Tensor): Continuous time parameter
            cond (torch.Tensor, optional): Conditioning. Defaults to None.

        Returns:
            torch.Tensor: Predicted noise
        """
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - expand_dims(alpha_t, x.dim()) * output) / expand_dims(sigma_t, x.dim())
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return expand_dims(alpha_t, x.dim()) * output + expand_dims(sigma_t, x.dim()) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output

    def cond_grad_fn(x, t_input):
        """Compute classifier gradient for classifier guidance.

        Args:
            x (torch.Tensor): Input tensor
            t_input (torch.Tensor): Time parameter in model's format

        Returns:
            torch.Tensor: Gradient of classifier log probability
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """Main model function for DPM-Solver.

        Handles different guidance types and applies appropriate guidance calculations.

        Args:
            x (torch.Tensor): Input tensor
            t_continuous (torch.Tensor): Continuous time parameter

        Returns:
            torch.Tensor: Predicted noise, modified by guidance if applicable
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                if isinstance(condition, torch.Tensor):
                    c_in = torch.cat([unconditional_condition, condition])
                elif isinstance(condition, dict):
                    c_in = {
                        k: torch.cat([unconditional_condition[k], condition[k]])
                        for k in condition
                    }
                else:
                    raise TypeError(f"classifier free guidance doesn't support condition with type {type(condition)}")
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                if not cfg_rescale:
                    return noise_uncond + guidance_scale * (noise - noise_uncond)
                else:
                    t_input = get_model_input_time(t_continuous)
                    cfg_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((1000 - t_input) / 1000) ** 5.0)) / 2
                    )
                    return noise_uncond + cfg_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        """Initialize a DPM-Solver instance.

        DPM-Solver is a fast dedicated high-order solver for diffusion ODEs with the convergence order guarantee.
        This class implements both DPM-Solver and DPM-Solver++ algorithms.

        Parameters
        ----------
        model_fn : callable
            A noise prediction model function that accepts continuous-time input.
            Should have signature: model_fn(x: Tensor, t_continuous: Tensor) -> Tensor
            where x has shape (batch_size, *data_shape) and t_continuous has shape (batch_size,)
        
        noise_schedule : object
            A noise schedule object (e.g. NoiseScheduleVP) that defines the diffusion process
        
        algorithm_type : str, optional
            The type of solver algorithm to use, by default "dpmsolver++"
            Must be one of ["dpmsolver", "dpmsolver++"]
        
        correcting_x0_fn : callable or str, optional
            Function to correct x0 predictions at each step, by default None
            If str "dynamic_thresholding", uses dynamic thresholding from Imagen paper
            If callable, should have signature: correcting_x0_fn(x0: Tensor, t: Tensor) -> Tensor
        
        correcting_xt_fn : callable, optional
            Function to correct intermediate samples xt at each step, by default None
            Should have signature: correcting_xt_fn(xt: Tensor, t: Tensor, step: int) -> Tensor
        
        thresholding_max_val : float, optional
            Maximum value for dynamic thresholding, by default 1.0
            Only used when algorithm_type="dpmsolver++" and correcting_x0_fn="dynamic_thresholding"
        
        dynamic_thresholding_ratio : float, optional
            Ratio for dynamic thresholding quantile computation, by default 0.995
            Only used when algorithm_type="dpmsolver++" and correcting_x0_fn="dynamic_thresholding"

        Notes
        -----
        The dynamic thresholding method from Imagen [1] can improve sample quality for pixel-space
        diffusion models with large guidance scales. However, it is unsuitable for latent-space
        diffusion models like Stable Diffusion.

        References
        ----------
        .. [1] Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language
           Understanding", https://arxiv.org/abs/2205.11487
        """
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """Apply dynamic thresholding to predicted x0.

        Parameters
        ----------
        x0 : torch.Tensor
            The predicted x0 values
        t : torch.Tensor
            The current timestep

        Returns
        -------
        torch.Tensor
            The thresholded x0 values

        Notes
        -----
        Implements dynamic thresholding from Imagen paper. Computes percentile of absolute
        values and clamps predictions to that threshold, then rescales to [-1,1].
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """Get noise prediction from the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        t : torch.Tensor
            Time parameter

        Returns
        -------
        torch.Tensor
            Predicted noise
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """Get data (x0) prediction from noise prediction model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        t : torch.Tensor
            Time parameter

        Returns
        -------
        torch.Tensor
            Predicted clean data x0, optionally corrected by correcting_x0_fn
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """Convert model output based on algorithm type.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        t : torch.Tensor
            Time parameter

        Returns
        -------
        torch.Tensor
            Model prediction (noise for dpmsolver, data for dpmsolver++)
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Parameters
        ----------
        skip_type : str
            The type for spacing of time steps. Supported types:
            - 'logSNR': uniform logSNR for time steps
            - 'time_uniform': uniform time steps (recommended for high-resolution data)
            - 'time_quadratic': quadratic time steps (used in DDIM for low-resolution data)
        t_T : float
            Starting time of sampling (default is T)
        t_0 : float 
            Ending time of sampling (default is epsilon)
        N : int
            Total number of time step intervals
        device : torch.device
            Device to place tensors on

        Returns
        -------
        torch.Tensor
            Time steps tensor with shape (N + 1,)
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """Get order of each step for sampling with singlestep DPM-Solver.

        Combines DPM-Solver-1,2,3 to use all function evaluations, called "DPM-Solver-fast".
        Given fixed number of function evaluations (steps), the sampling procedure is:

        - If order == 1:
            Takes `steps` of DPM-Solver-1 (i.e. DDIM)
        - If order == 2:
            - Let K = (steps // 2). Takes K or (K + 1) intermediate time steps
            - If steps % 2 == 0: Uses K steps of DPM-Solver-2
            - If steps % 2 == 1: Uses K steps of DPM-Solver-2 and 1 step of DPM-Solver-1
        - If order == 3:
            - Let K = (steps // 3 + 1). Takes K intermediate time steps
            - If steps % 3 == 0: Uses (K-2) steps of DPM-Solver-3, 1 step of DPM-Solver-2, 1 step of DPM-Solver-1
            - If steps % 3 == 1: Uses (K-1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1
            - If steps % 3 == 2: Uses (K-1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2

        Parameters
        ----------
        steps : int
            Total number of function evaluations (NFE)
        order : int
            Maximum solver order (1, 2 or 3)
        skip_type : str
            Time step spacing type:
            - 'logSNR': uniform logSNR spacing
            - 'time_uniform': uniform time spacing (recommended for high-res data)
            - 'time_quadratic': quadratic time spacing (for DDIM low-res data)
        t_T : float
            Starting time of sampling (default is T)
        t_0 : float
            Ending time of sampling (default is epsilon)
        device : torch.device
            Device to place tensors on

        Returns
        -------
        tuple
            (timesteps_outer, orders) where:
            - timesteps_outer: tensor of outer loop timesteps
            - orders: list of solver orders for each step
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """Denoise at final step by solving ODE from lambda_s to infinity.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to denoise
        s : torch.Tensor
            Time parameter

        Returns
        -------
        torch.Tensor
            Denoised prediction
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """DPM-Solver-1 update (equivalent to DDIM) from time s to t.

        Parameters
        ----------
        x : torch.Tensor
            Initial value at time s
        s : torch.Tensor
            Starting time with shape (1,)
        t : torch.Tensor
            Ending time with shape (1,)
        model_s : torch.Tensor, optional
            Model function evaluated at time s. If None, evaluates model at (x,s)
        return_intermediate : bool, optional
            If True, also returns model value at time s

        Returns
        -------
        torch.Tensor or tuple
            - If return_intermediate=False: approximated solution at time t
            - If return_intermediate=True: tuple (solution, dict with intermediate values)
        """
        ns = self.noise_schedule
        dims = x.dim()
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                (sigma_s1 / sigma_s) * x
                - (alpha_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r1) * (alpha_t * (phi_1 / h + 1.)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == 'taylor':
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r1) * (sigma_t * (phi_1 / h - 1.)) * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        """Implements single-step third-order DPM-Solver update from time s to t.

        This method implements a third-order update step for DPM-Solver, which provides higher accuracy
        than first or second order methods by using two intermediate evaluations.

        Args:
            x (torch.Tensor): The initial value at time s.
            s (torch.Tensor): The starting time with shape (1,).
            t (torch.Tensor): The ending time with shape (1,).
            r1 (float, optional): First hyperparameter controlling intermediate step locations.
                Default: 1/3.
            r2 (float, optional): Second hyperparameter controlling intermediate step locations.
                Default: 2/3.
            model_s (torch.Tensor, optional): The model output at time s. If None, will be computed
                from x and s. Default: None.
            model_s1 (torch.Tensor, optional): The model output at first intermediate time s1.
                If None, will be computed. Default: None.
            return_intermediate (bool, optional): If True, also returns model outputs at intermediate
                timesteps. Default: False.
            solver_type (str, optional): The type of solver to use - either 'dpmsolver' or 'taylor'.
                Impacts numerical performance. Default: 'dpmsolver'.

        Returns:
            torch.Tensor or tuple: If return_intermediate is False, returns the approximated solution
            at time t. If return_intermediate is True, returns a tuple (x_t, intermediate_dict) where
            intermediate_dict contains the model outputs at times s, s1 and s2.

        Raises:
            ValueError: If solver_type is not 'dpmsolver' or 'taylor'.

        Notes:
            - The method uses either DPM-Solver++ or regular DPM-Solver algorithm based on
              self.algorithm_type
            - For DPM-Solver++, the update uses noise prediction parameterization
            - For regular DPM-Solver, the update uses data prediction parameterization
            - The intermediate times s1 and s2 are computed based on r1 and r2 parameters
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    - (alpha_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1. / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    (torch.exp(log_alpha_s1 - log_alpha_s)) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1. / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == 'taylor':
                D1_0 = (1. / r1) * (model_s1 - model_s)
                D1_1 = (1. / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2. * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """Implements multistep second-order DPM-Solver update.

        This method implements a second-order multistep update for DPM-Solver, using the model
        outputs from previous steps to achieve higher accuracy.

        Args:
            x (torch.Tensor): The initial value at time t_prev_list[-1].
            model_prev_list (list of torch.Tensor): List of previous model outputs, ordered from oldest
                to newest. Should contain exactly 2 elements.
            t_prev_list (list of torch.Tensor): List of previous timesteps corresponding to
                model_prev_list. Each tensor has shape (1,). Should contain exactly 2 elements.
            t (torch.Tensor): The target time to update to, with shape (1,).
            solver_type (str, optional): The type of solver to use - either 'dpmsolver' or 'taylor'.
                Impacts numerical performance. Default: 'dpmsolver'.

        Returns:
            torch.Tensor: The approximated solution at time t.

        Raises:
            ValueError: If solver_type is not 'dpmsolver' or 'taylor'.

        Notes:
            - The method uses either DPM-Solver++ or regular DPM-Solver algorithm based on
              self.algorithm_type
            - For DPM-Solver++, the update uses noise prediction parameterization
            - For regular DPM-Solver, the update uses data prediction parameterization
            - The method requires exactly two previous model evaluations
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    - 0.5 * (alpha_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == 'taylor':
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.)) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """Multistep solver DPM-Solver-3 from time t_prev_list[-1] to time t.

        This method implements the third-order multistep DPM-Solver update. It uses three previous model
        evaluations to compute a more accurate solution at the target time t.

        Parameters
        ----------
        x : torch.Tensor
            The initial value at time t_prev_list[-1]
        model_prev_list : list of torch.Tensor
            List containing three previous model evaluations [model(t-2), model(t-1), model(t)]
        t_prev_list : list of torch.Tensor
            List containing three previous timesteps [t-2, t-1, t], each with shape (1,)
        t : torch.Tensor
            The target time to update to, with shape (1,)
        solver_type : str, optional
            The type of solver to use, either 'dpmsolver' or 'taylor'. Default: 'dpmsolver'
            The solver type slightly impacts performance, with 'dpmsolver' recommended.

        Returns
        -------
        torch.Tensor
            The approximated solution at time t

        Notes
        -----
        - The method uses either DPM-Solver++ or regular DPM-Solver algorithm based on
          self.algorithm_type
        - For DPM-Solver++, the update uses noise prediction parameterization
        - For regular DPM-Solver, the update uses data prediction parameterization
        - The method requires exactly three previous model evaluations

        References
        ----------
        Algorithm 1 and 2 in https://arxiv.org/abs/2206.00927
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """Singlestep DPM-Solver update from time s to time t.

        Parameters
        ----------
        x : torch.Tensor
            The initial value at time s
        s : torch.Tensor
            The starting time, with shape (1,)
        t : torch.Tensor
            The ending time, with shape (1,)
        order : int
            The order of DPM-Solver (1, 2, or 3)
        return_intermediate : bool, optional
            If True, also return model values at intermediate times. Default: False
        solver_type : str, optional
            The solver type ('dpmsolver' or 'taylor'). Default: 'dpmsolver'
        r1 : float, optional
            Hyperparameter for second/third-order solver
        r2 : float, optional
            Hyperparameter for third-order solver

        Returns
        -------
        torch.Tensor or tuple
            If return_intermediate is False, returns approximated solution at time t
            If True, returns tuple with solution and intermediate values

        Raises
        ------
        ValueError
            If order is not 1, 2, or 3
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """Multistep DPM-Solver update from time t_prev_list[-1] to time t.

        Parameters
        ----------
        x : torch.Tensor
            The initial value at time t_prev_list[-1]
        model_prev_list : list of torch.Tensor
            Previous computed model values
        t_prev_list : list of torch.Tensor
            Previous times, each with shape (1,)
        t : torch.Tensor
            The ending time, with shape (1,)
        order : int
            The order of DPM-Solver (1, 2, or 3)
        solver_type : str, optional
            The solver type ('dpmsolver' or 'taylor'). Default: 'dpmsolver'

        Returns
        -------
        torch.Tensor
            The approximated solution at time t

        Raises
        ------
        ValueError
            If order is not 1, 2, or 3
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        """Adaptive step size solver based on singlestep DPM-Solver.

        This implements an adaptive step size solver that automatically adjusts the step size
        based on estimated local error to maintain accuracy while maximizing efficiency.

        Parameters
        ----------
        x : torch.Tensor
            The initial value at time t_T
        order : int
            The (higher) order of the solver (2 or 3)
        t_T : float
            The starting time of sampling
        t_0 : float
            The ending time of sampling
        h_init : float, optional
            Initial step size (for logSNR). Default: 0.05
        atol : float, optional
            Absolute tolerance. Default: 0.0078
        rtol : float, optional
            Relative tolerance. Default: 0.05
        theta : float, optional
            Safety factor for step size adaptation. Default: 0.9
        t_err : float, optional
            Time tolerance for ODE solution. Default: 1e-5
        solver_type : str, optional
            The solver type ('dpmsolver' or 'taylor'). Default: 'dpmsolver'

        Returns
        -------
        torch.Tensor
            The approximated solution at time t_0

        Notes
        -----
        The adaptive step size algorithm follows:
        1. Compute solutions with both lower and higher order methods
        2. Estimate local error by comparing solutions
        3. Accept step if error is within tolerance
        4. Adjust step size based on error estimate

        References
        ----------
        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas,
        "Gotta go fast when generating data with score-based models," 
        arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """Add noise to input data according to the noise schedule.

        Computes the noised input xt = alpha_t * x + sigma_t * noise, where alpha_t and sigma_t
        are determined by the noise schedule at timestep t.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, *shape).
        t : torch.Tensor 
            Time steps tensor with shape (t_size,).
        noise : torch.Tensor, optional
            Optional noise tensor with shape (t_size, batch_size, *shape). If None, random
            Gaussian noise will be generated.

        Returns
        -------
        torch.Tensor
            Noised input tensor with shape (t_size, batch_size, *shape) if t_size > 1,
            or (batch_size, *shape) if t_size == 1.

        Notes
        -----
        The noise is added according to:
            xt = alpha_t * x + sigma_t * noise
        where:
            - alpha_t is the scaling factor at time t
            - sigma_t is the noise level at time t
            - noise is sampled from N(0,1) if not provided
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """Invert the sampling process from time t_start to t_end.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to invert.
        steps : int, optional
            Number of solver steps, by default 20
        t_start : float, optional
            Starting time. If None, defaults to 1/N where N is total timesteps.
        t_end : float, optional  
            Ending time. If None, defaults to T (usually 1.0).
        order : int, optional
            Order of the solver (1, 2, or 3), by default 2
        skip_type : str, optional
            Time step spacing ('time_uniform', 'logSNR', 'time_quadratic'), by default 'time_uniform'
        method : str, optional
            Solver method ('multistep', 'singlestep', 'adaptive'), by default 'multistep'
        lower_order_final : bool, optional
            Whether to use lower order solvers for final steps, by default True
        denoise_to_zero : bool, optional
            Whether to denoise to t=0 at final step, by default False
        solver_type : str, optional
            Type of solver ('dpmsolver' or 'taylor'), by default 'dpmsolver'
        atol : float, optional
            Absolute tolerance for adaptive solver, by default 0.0078
        rtol : float, optional
            Relative tolerance for adaptive solver, by default 0.05
        return_intermediate : bool, optional
            Whether to return intermediate samples, by default False

        Returns
        -------
        torch.Tensor
            The inverted sample at time t_end.

        Notes
        -----
        For discrete-time DPMs, t_start should be 1/N where N is total timesteps.
        This is a wrapper around sample() that inverts the time direction.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """Sample from the model using DPM-Solver.

        Parameters
        ----------
        x : torch.Tensor
            Initial value at time t_start.
        steps : int, optional
            Number of solver steps, by default 20
        t_start : float, optional
            Starting time. If None, defaults to T (usually 1.0).
        t_end : float, optional
            Ending time. If None, defaults to 1/N where N is total timesteps.
        order : int, optional
            Order of the solver (1, 2, or 3), by default 2
        skip_type : str, optional
            Time step spacing ('time_uniform', 'logSNR', 'time_quadratic'), by default 'time_uniform'
        method : str, optional
            Solver method ('multistep', 'singlestep', 'adaptive'), by default 'multistep'
        lower_order_final : bool, optional
            Whether to use lower order solvers for final steps, by default True
        denoise_to_zero : bool, optional
            Whether to denoise to t=0 at final step, by default False
        solver_type : str, optional
            Type of solver ('dpmsolver' or 'taylor'), by default 'dpmsolver'
        atol : float, optional
            Absolute tolerance for adaptive solver, by default 0.0078
        rtol : float, optional
            Relative tolerance for adaptive solver, by default 0.05
        return_intermediate : bool, optional
            Whether to return intermediate samples, by default False

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]
            If return_intermediate is False, returns the final sample tensor.
            If return_intermediate is True, returns a tuple of (final sample, list of intermediate samples).

        Notes
        -----
        Supported algorithms:
        - 'singlestep': Combines different orders of singlestep DPM-Solver
        - 'multistep': Multistep DPM-Solver with specified order
        - 'singlestep_fixed': Fixed order singlestep DPM-Solver
        - 'adaptive': Adaptive step size DPM-Solver

        For best results:
        - Unconditional/lightly guided sampling: Use singlestep order=3
        - Heavily guided sampling: Use multistep order=2
        """
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        # intermediates.append(x)
                        intermediates.append(self.data_prediction_fn(x, t))
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        # intermediates.append(x)
                        intermediates.append(self.data_prediction_fn(x, t))
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x



#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """Piecewise linear interpolation function y = f(x) using keypoints.

    Implements a differentiable piecewise linear interpolation function using xp and yp as keypoints.
    The function is well-defined for all x values - for x beyond the bounds of xp, uses the outermost
    points of xp to define the linear function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [N, C], where:
            N is the batch size
            C is the number of channels (typically 1 for DPM-Solver)
    xp : torch.Tensor 
        X-coordinates of keypoints, shape [C, K], where:
            K is the number of keypoints
    yp : torch.Tensor
        Y-coordinates of keypoints, shape [C, K]

    Returns
    -------
    torch.Tensor
        Interpolated values f(x) with shape [N, C]

    Notes
    -----
    The function is implemented to be differentiable and work with autograd.
    For x values outside the range of xp, extrapolates using the outermost line segments.
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """Expand a tensor to a specified number of dimensions.

    Parameters
    ----------
    v : torch.Tensor
        Input tensor with shape [N]
    dims : int
        Target number of dimensions

    Returns
    -------
    torch.Tensor
        Expanded tensor with shape [N, 1, 1, ..., 1] where the total number
        of dimensions equals `dims`

    Examples
    --------
    >>> v = torch.tensor([1, 2, 3])
    >>> expand_dims(v, 3)
    tensor([[[1],
             [2], 
             [3]]])
    """
    return v[(...,) + (None,)*(dims - 1)]