import torch
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels[0], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[1], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels[1], 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, output_channels[2], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels[2], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[3], kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[4], kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU()
        )
        
        self.block_6 = torch.nn.Sequential(
            torch.nn.Conv2d(output_channels[4], 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[5], kernel_size=3, stride=1, padding=0)
        )
        
        self.blocks = [self.block_1, self.block_2, self.block_3, self.block_4, self.block_5, self.block_6]
        

    def forward(self, x):
        """
        The forward function should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
             shape(-1, output_channels[1], 19, 19),
             shape(-1, output_channels[2], 10, 10),
             shape(-1, output_channels[3], 5, 5),
             shape(-1, output_channels[3], 3, 3),
             shape(-1, output_channels[4], 1, 1)]
        """
        out_features = []
        for block in self.blocks:
            out_features.append(block(x))
            x = out_features[-1]
            
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)
