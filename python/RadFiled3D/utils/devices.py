from typing import Any

import torch


def move_nested_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=obj.is_pinned())
    elif isinstance(obj, dict):
        return {k: move_nested_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return type(obj)(move_nested_to_device(item, device) for item in obj)
    elif hasattr(obj, "_fields"):  # NamedTuple
        return type(obj)(
            *(
                move_nested_to_device(getattr(obj, field), device)
                for field in obj._fields
            )
        )
    else:
        return obj
