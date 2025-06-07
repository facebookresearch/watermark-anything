# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import subprocess
from typing import Dict

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def is_distributed():
    return get_world_size() > 1

def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import sys
    import builtins as __builtin__

    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    builtin_stderr_write = sys.stderr.write
    def stderr_write(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_stderr_write(*args)

    sys.stderr.write = stderr_write

def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    The params need to contain the following attributes:
        - local_rank
        - master_port
        - debug_slurm
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm
    print("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        if params.master_port==-1:
            params.master_port = '19500'
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        print(PREFIX + "Master address: %s" % params.master_addr)
        print(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch or torchrun
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        # params.global_rank = int(os.environ['RANK'])
        # params.world_size = int(os.environ['WORLD_SIZE'])
        # params.n_gpu_per_node = int(os.environ['NGPU'])

        # # number of nodes / node ID
        # params.n_nodes = params.world_size // params.n_gpu_per_node
        # params.node_id = params.global_rank // params.n_gpu_per_node
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.local_rank = int(os.environ['LOCAL_RANK'])
        params.n_gpu_per_node = int(os.environ['CUDA_VISIBLE_DEVICES'].count(',') + 1)
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.distributed = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank
    print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    print(PREFIX + "Node ID        : %i" % params.node_id)
    print(PREFIX + "Local rank     : %i" % params.local_rank)
    print(PREFIX + "Global rank    : %i" % params.global_rank)
    print(PREFIX + "World size     : %i" % params.world_size)
    print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(params.is_master))
    print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(params.distributed))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # initialize multi-GPU
    if params.distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
            world_size=params.world_size,
            rank=params.global_rank
        )

        # set GPU device
        torch.cuda.set_device(params.local_rank)
        # dist.barrier(device_ids=[params.local_rank])
        setup_for_distributed(params.is_master)

def average_metrics(metrics: Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))
