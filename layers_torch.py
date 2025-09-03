import torch
import torch.nn as nn
import torch.nn.functional as F

class AllViewsGaussianNoise(nn.Module):
    def __init__(self, gaussian_noise_std, device):
        super().__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.device = device

    def forward(self, x):
        if not self.gaussian_noise_std:
            return x
        return {view: self._add_gaussian_noise(img) for view, img in x.items()}

    def _add_gaussian_noise(self, single_view):
        noise = torch.randn_like(single_view) * self.gaussian_noise_std
        return single_view + noise.to(self.device)

class AllViewsConvLayer(nn.Module):
    def __init__(self, in_channels, number_of_filters=32,
                 filter_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.cc = nn.Conv2d(in_channels, number_of_filters,
                           filter_size, stride)
        self.mlo = nn.Conv2d(in_channels, number_of_filters,
                            filter_size, stride)

    def forward(self, x):
        return {
            view: F.relu(self.cc(img) if "CC" in view else self.mlo(img))
            for view, img in x.items()
        }

    @property
    def ops(self):
        return {"CC": self.cc, "MLO": self.mlo}

class AllViewsMaxPool(nn.Module):
    def forward(self, x, stride=(2, 2), padding=(0, 0)):
        return {
            view: F.max_pool2d(img, stride, stride, padding)
            for view, img in x.items()
        }

class AllViewsAvgPool(nn.Module):
    def forward(self, x):
        return {view: self._avg_pool(img) for view, img in x.items()}

    @staticmethod
    def _avg_pool(single_view):
        n, c, h, w = single_view.size()
        return single_view.view(n, c, -1).mean(-1)

class AllViewsPad(nn.Module):
    def forward(self, x, pad):
        return {view: F.pad(img, pad) for view, img in x.items()}