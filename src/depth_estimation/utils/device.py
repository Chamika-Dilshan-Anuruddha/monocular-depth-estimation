""" CPU / CUDA/ MPS device detection with frindly logging."""

from __future__ import annotations

import torch

from depth_estimation.utils import get_logger


log = get_logger(__name__)

def get_device(prefer: str | None = None) -> torch.device:

    if prefer: 
        device = torch.device(prefer)
        log.info("Device forced by caller: %s", device)
        return device
    
    if torch.cuda.is_available():
        device  = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("CUDA device selected: %s (%.1f GB VRAM)",name, mem_gb)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Apple MPS device selected")
    else:
        device = torch.device("cpu")
        log.info("Falling back to CPU")
    
    return device


def device_info() -> dict[str, str | bool]:
    
    cuda_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    info: dict [str, str | bool] = {
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "mps_available": mps_avail,
        "selected_device":str(get_device())
    }

    if cuda_avail:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "unknown"
    
    return info


    
