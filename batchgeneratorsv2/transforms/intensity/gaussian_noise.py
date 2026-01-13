import os
from typing import Tuple
import torch
import numpy as np
import os
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class GaussianNoiseTransform(ImageOnlyTransform):
    def __init__(self,
                 noise_variance: RandomScalar = (0, 0.1),
                 p_per_channel: float = 1.,
                 synchronize_channels: bool = False):
        super().__init__()
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.synchronize_channels = synchronize_channels

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        c = img.shape[0]

        # bool mask on same device as image
        apply_np = np.random.rand(c) < self.p_per_channel
        apply = torch.from_numpy(apply_np).to(device=img.device)

        # store also count / indices to avoid recomputing later
        idx = apply.nonzero(as_tuple=False).flatten()
        n = idx.numel()

        if n == 0:
            sigmas = None
        elif self.synchronize_channels:
            sigmas = sample_scalar(self.noise_variance, img)
        else:
            # still uses sample_scalar, but avoids list->cat in _apply
            # if sample_scalar is cheap, this is fine; otherwise see note below
            sigmas = [sample_scalar(self.noise_variance, img) for _ in range(n)]

        return {"apply_mask": apply, "apply_idx": idx, "num_apply": n, "sigmas": sigmas}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        n = params["num_apply"]
        if n == 0:
            return img

        idx = params["apply_idx"]
        spatial = img.shape[1:]
        device = img.device
        dtype = img.dtype

        sigmas = params["sigmas"]

        if sigmas is None:
            return img

        # Create noise only for selected channels using NumPy
        if not self.synchronize_channels:
            # Generate noise per channel with different sigma values
            noise_shape = (n, *spatial)
            noise_np = np.random.randn(*noise_shape).astype(np.float32)
            
            # Apply per-channel sigma scaling
            for i, sigma in enumerate(sigmas):
                noise_np[i] *= sigma
            
            noise = torch.from_numpy(noise_np).to(device=device, dtype=dtype)
        else:
            sigma = sigmas
            noise_shape = (n, *spatial)
            noise_np = np.random.normal(loc=0.0, scale=float(sigma), size=noise_shape).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(device=device, dtype=dtype)

        # In-place add only on selected channels
        img[idx].add_(noise)
        return img
    

if __name__ == "__main__":
    from time import time
    import numpy as np

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    gnt = GaussianNoiseTransform((0, 0.1), 1, False)

    times = []
    for _ in range(1000):
        data_dict = {'image': torch.ones((2, 32, 32, 32))}
        st = time()
        out = gnt(**data_dict)
        times.append(time() - st)
    print('torch', np.mean(times))

    from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

    gnt_bg = GaussianNoiseTransform((0, 0.1), 1, 1, True)

    times = []
    for _ in range(1000):
        data_dict = {'data': np.ones((1, 2, 32, 32, 32))}
        st = time()
        out = gnt_bg(**data_dict)
        times.append(time() - st)

    print('bg', np.mean(times))
    # torch is 2.5x faster
