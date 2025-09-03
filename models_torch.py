import torch
import torch.nn as nn
import torch.nn.functional as F
import collections as col
import layers_torch as layers


class BaselineBreastModel(nn.Module):
    def __init__(self, device, nodropout_probability=None, gaussian_noise_std=None):
        super().__init__()
        self.conv_layer_dict = self._build_conv_layers()
        self._conv_layer_ls = nn.ModuleList(self.conv_layer_dict.values())

        # Additional layers
        self.all_views_max_pool = layers.AllViewsMaxPool()
        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_pad = layers.AllViewsPad()
        self.gaussian_noise_layer = layers.AllViewsGaussianNoise(
            gaussian_noise_std, device)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4, 256 * 4)
        self.fc2 = nn.Linear(256 * 4, 3)
        self.dropout = nn.Dropout(p=1 - nodropout_probability)

    def _build_conv_layers(self):
        conv_layers = col.OrderedDict()

        # First conv sequence
        conv_layers["conv1"] = layers.AllViewsConvLayer(
            1, 32, filter_size=(3, 3), stride=(2, 2))

        # Second conv sequence
        conv_layers["conv2a"] = layers.AllViewsConvLayer(
            32, 64, filter_size=(3, 3), stride=(2, 2))
        conv_layers["conv2b"] = layers.AllViewsConvLayer(
            64, 64, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv2c"] = layers.AllViewsConvLayer(
            64, 64, filter_size=(3, 3), stride=(1, 1))

        # Third conv sequence
        conv_layers["conv3a"] = layers.AllViewsConvLayer(
            64, 128, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv3b"] = layers.AllViewsConvLayer(
            128, 128, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv3c"] = layers.AllViewsConvLayer(
            128, 128, filter_size=(3, 3), stride=(1, 1))

        # Fourth conv sequence
        conv_layers["conv4a"] = layers.AllViewsConvLayer(
            128, 128, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv4b"] = layers.AllViewsConvLayer(
            128, 128, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv4c"] = layers.AllViewsConvLayer(
            128, 128, filter_size=(3, 3), stride=(1, 1))

        # Fifth conv sequence
        conv_layers["conv5a"] = layers.AllViewsConvLayer(
            128, 256, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv5b"] = layers.AllViewsConvLayer(
            256, 256, filter_size=(3, 3), stride=(1, 1))
        conv_layers["conv5c"] = layers.AllViewsConvLayer(
            256, 256, filter_size=(3, 3), stride=(1, 1))

        return conv_layers

    def forward(self, x):
        x = self.gaussian_noise_layer(x)

        # First conv sequence
        x = self.conv_layer_dict["conv1"](x)

        # Second conv sequence
        x = self.all_views_max_pool(x, stride=(3, 3))
        x = self.conv_layer_dict["conv2a"](x)
        x = self.conv_layer_dict["conv2b"](x)
        x = self.conv_layer_dict["conv2c"](x)

        # Third conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv3a"](x)
        x = self.conv_layer_dict["conv3b"](x)
        x = self.conv_layer_dict["conv3c"](x)

        x = self.all_views_pad(x, pad=(0, 1, 0, 0))

        # Fourth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv4a"](x)
        x = self.conv_layer_dict["conv4b"](x)
        x = self.conv_layer_dict["conv4c"](x)

        # Fifth conv sequence
        x = self.all_views_max_pool(x, stride=(2, 2))
        x = self.conv_layer_dict["conv5a"](x)
        x = self.conv_layer_dict["conv5b"](x)
        x = self.conv_layer_dict["conv5c"](x)

        # Final processing
        x = self.all_views_avg_pool(x)
        x = torch.cat([x[view] for view in ["L-CC", "R-CC", "L-MLO", "R-MLO"]],
                      dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)