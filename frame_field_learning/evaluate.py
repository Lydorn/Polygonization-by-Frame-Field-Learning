import random
import torch
import torch.distributed
import torch.utils.data

from . import data_transforms
from .model import FrameFieldModel
from .evaluator import Evaluator

from lydorn_utils import print_utils

try:
    import apex
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def evaluate(gpu: int, config: dict, shared_dict, barrier, eval_ds, backbone):
    # --- Setup DistributedDataParallel --- #
    rank = config["nr"] * config["gpus"] + gpu
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=config["world_size"],
        rank=rank
    )

    if gpu == 0:
        print("# --- Start evaluating --- #")

    # Choose device
    torch.cuda.set_device(gpu)

    # --- Online transform performed on the device (GPU):
    eval_online_cuda_transform = data_transforms.get_eval_online_cuda_transform(config)

    if "samples" in config:
        rng_samples = random.Random(0)
        eval_ds = torch.utils.data.Subset(eval_ds, rng_samples.sample(range(len(eval_ds)), config["samples"]))
        # eval_ds = torch.utils.data.Subset(eval_ds, range(config["samples"]))

    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_ds, num_replicas=config["world_size"], rank=rank)

    eval_ds = torch.utils.data.DataLoader(eval_ds, batch_size=config["optim_params"]["eval_batch_size"], pin_memory=True, sampler=eval_sampler, num_workers=config["num_workers"])

    model = FrameFieldModel(config, backbone=backbone, eval_transform=eval_online_cuda_transform)
    model.cuda(gpu)

    if config["use_amp"] and APEX_AVAILABLE:
        amp.register_float_function(torch, 'sigmoid')
        model = amp.initialize(model, opt_level="O1")
    elif config["use_amp"] and not APEX_AVAILABLE and gpu == 0:
        print_utils.print_warning("WARNING: Cannot use amp because the apex library is not available!")

    # Wrap the model for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    evaluator = Evaluator(gpu, config, shared_dict, barrier, model, run_dirpath=config["eval_params"]["run_dirpath"])
    split_name = config["fold"][0]
    evaluator.evaluate(split_name, eval_ds)
