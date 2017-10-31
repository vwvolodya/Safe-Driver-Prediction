import torch


class ToTensor:
    def __call__(self, x):
        result = {k: torch.from_numpy(v).float() for k, v in x.items()}
        return result
