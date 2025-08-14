import torch
from torchdiffeq import odeint

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.flatten(1).mean(1)

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(torch.pi / 2 * t) + 1))

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

def rf_training_losses(model, x_start, model_kwargs=None, noise=None, f0_channels=32, predict="flow"):
    """
    Compute training losses for a single timestep.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if noise is None:
        noise = torch.randn_like(x_start)
    
    times = torch.rand(x_start.shape[0], device=x_start.device)
    padded_times = append_dims(times, x_start.ndim - 1)
    
    t = cosmap(padded_times)
    x_t = t * x_start + (1. - t) * noise
    
    x_t = torch.cat([x_start[:, :, :f0_channels], x_t[:, :, f0_channels:]], dim=-1)
    
    flow = x_start - noise

    terms = {}
    
    model_output = model(x_t, t.squeeze(-1).squeeze(-1), **model_kwargs)

    if predict == 'flow':
        target = flow
    elif predict == 'noise':
        target = noise
    else:
        raise ValueError(f'unknown objective {predict}')
    
    ft_channels = x_start.shape[-1] - f0_channels
        
    terms["mse"] = mean_flat((target[:, :, -ft_channels:] - model_output[:, :, -ft_channels:]) ** 2)

    terms["loss"] = terms["mse"]

    return terms

@torch.no_grad()
def rf_sample(
    model,
    shape,
    steps=64,
    model_kwargs=None,
    device=None,
    guidance_scale=3.0,
    predict='flow',
    f0=None,
):   
    
    f0_channels = f0.shape[-1]
    
    odeint_kwargs = dict(
        atol = 1e-5,
        rtol = 1e-5,
        method = 'midpoint'
    )
    
    model_kwargs['texts'].append('')

    def ode_fn(t, x):
        x = torch.cat([x] * 2)
        
        x = torch.cat([f0.repeat(x.shape[0], 1, 1), x[:, :, f0_channels:]], dim=-1)
        if predict == 'flow':
            flow = model(x, t.unsqueeze(0).repeat(x.shape[0]), **model_kwargs)
        elif predict == 'noise':
            noise = model(x, t.unsqueeze(0).repeat(x.shape[0]), **model_kwargs)
            padded_times = append_dims(t, noised.ndim - 1)
            flow = (noised - noise) / padded_times.clamp(min = 1e-10)
        else:
            raise ValueError(f'unknown objective {predict}')

        # cfg
        cond_flow, uncond_flow = torch.split(flow, len(flow) // 2, dim=0)
        flow = uncond_flow + guidance_scale * (cond_flow - uncond_flow)

        return flow

    noise = torch.randn(*shape, device=device) 
    times = torch.linspace(0., 1., steps, device=device)

    # ode
    trajectory = odeint(ode_fn, noise, times, **odeint_kwargs)
    
    sampled_data = trajectory[-1][:1]

    sampled_data = torch.cat([f0, sampled_data[:, :, f0_channels:]], dim=-1)
     
    return sampled_data