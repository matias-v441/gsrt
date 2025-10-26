import torch

class Rays:
    def __init__(self, *, origins: torch.Tensor, directions: torch.Tensor, res_x: int, res_y: int):
        self.origins = origins
        self.directions = directions
        self.res_x = res_x
        self.res_y = res_y

__all__ = ["Rays", "TraceFunction", "GaussianModel", "Activations", "Training"]