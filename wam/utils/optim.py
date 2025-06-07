# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math

import torch
import timm.optim as optim
import timm.scheduler as scheduler

def parse_params(s):
    """
    Parse parameters into a dictionary, used for optimizer and scheduler parsing.
    Example: 
        "SGD,lr=0.01" -> {"name": "SGD", "lr": 0.01}
    """
    s = s.replace(' ', '').split(',')
    params = {}
    params['name'] = s[0]
    for x in s[1:]:
        x = x.split('=')
        params[x[0]]=float(x[1])
    return params

def build_optimizer(
    name, 
    model_params, 
    **optim_params
) -> torch.optim.Optimizer:
    """ Build optimizer from a dictionary of parameters """
    tim_optimizers = sorted(name for name in optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(optim.__dict__[name]))
    torch_optimizers = sorted(name for name in torch.optim.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.__dict__[name]))
    if hasattr(optim, name):
        return getattr(optim, name)(model_params, **optim_params)
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model_params, **optim_params)
    raise ValueError(f'Unknown optimizer "{name}", choose among {str(tim_optimizers+torch_optimizers)}')

def build_lr_scheduler(
    name, 
    optimizer, 
    **lr_scheduler_params
) -> torch.optim.lr_scheduler._LRScheduler:
    """ 
    Build scheduler from a dictionary of parameters 
    Args:
        name: name of the scheduler
        optimizer: optimizer to be used with the scheduler
        params: dictionary of scheduler parameters
    Ex:
        CosineLRScheduler, optimizer {t_initial=50, cycle_mul=2, cycle_limit=3, cycle_decay=0.5, warmup_lr_init=1e-6, warmup_t=5}
    """
    if name == "None" or name == "none":
        return None
    tim_schedulers = sorted(name for name in scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(scheduler.__dict__[name]))
    torch_schedulers = sorted(name for name in torch.optim.lr_scheduler.__dict__
        if name[0].isupper() and not name.startswith("__")
        and callable(torch.optim.lr_scheduler.__dict__[name]))
    if hasattr(scheduler, name):
        return getattr(scheduler, name)(optimizer, **lr_scheduler_params)
    elif hasattr(torch.optim.lr_scheduler, name):
        return getattr(torch.optim.lr_scheduler, name)(optimizer, **lr_scheduler_params)
    raise ValueError(f'Unknown scheduler "{name}", choose among {str(tim_schedulers+torch_schedulers)}')

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=True)
                except:
                    checkpoint[key] = {k.replace("module.", ""): v for k, v in checkpoint[key].items()}
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # test the optimizer
    params = parse_params("SGD,lr=0.01")
    print(params)
    model_params = torch.nn.Linear(10, 10).parameters()
    optimizer = build_optimizer(**params, model_params=model_params)
    print(optimizer)

    # test the scheduler
    params = parse_params("CosineLRScheduler,t_initial=50,cycle_mul=2,cycle_limit=3,cycle_decay=0.5,warmup_lr_init=1e-6,warmup_t=5")
    print(params)
    lr_scheduler = build_lr_scheduler(optimizer=optimizer, **params)
    print(lr_scheduler)

    # test the schedules
    class Test:
        def __init__(self, scaling):
            self.scaling = scaling

    scaling_o = 0.3
    scaling_min = 0.1
    # create and save plots
    scaling_sched = ScalingScheduler(Test, "scaling", "linear", scaling_o, scaling_min, 100)
    print("Linear: ", [scaling_sched.step(ii) for ii in range(100)])
    plt.plot([scaling_sched.step(ii) for ii in range(100)], label="linear")
    scaling_sched = ScalingScheduler(Test, "scaling", "cosine", scaling_o, scaling_min, 100)
    print("Cosine: ", [scaling_sched.step(ii) for ii in range(100)])
    plt.plot([scaling_sched.step(ii) for ii in range(100)], label="cosine")
    plt.savefig("schedules.png")
