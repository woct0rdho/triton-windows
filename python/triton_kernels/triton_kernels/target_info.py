import torch
import triton

cached_capabilities = {}


def is_cuda():
    if "is_cuda" not in cached_capabilities:
        target = triton.runtime.driver.active.get_current_target()
        cached_capabilities["is_cuda"] = False if target is None else target.backend == "cuda"
    return cached_capabilities["is_cuda"]


def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    if "cuda" not in cached_capabilities:
        if torch.cuda.is_available():
            cached_capabilities["cuda"] = torch.cuda.get_device_capability()
        else:
            cached_capabilities["cuda"] = (0, 0)
    return cached_capabilities["cuda"] >= (major, minor)


def num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count
