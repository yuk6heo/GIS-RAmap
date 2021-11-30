
import numpy as np
import torch

from config import Config
Config = Config()


class Normalize_ApplymeanvarImage(object): # For ResNet
    def __init__(self, mean, var, change_channels=False):
        self.mean = mean
        self.var = var
        self.change_channels = change_channels

    def __call__(self, sample):
        for elem in sample.keys():
            if 'image' in elem:
                if self.change_channels:
                    sample[elem] = sample[elem][:, :, [2, 1, 0]]
                sample[elem] = sample[elem].astype(np.float32)/255.0
                sample[elem] = np.subtract(sample[elem], np.array(self.mean, dtype=np.float32))/np.array(self.var, dtype=np.float32)


        return sample

    def __str__(self):
        return 'SubtractMeanImage'+str(self.mean)



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            if len(tmp.shape)==3:
                tmp = tmp.transpose((2, 0, 1))
            elif len(tmp.shape)==4:
                tmp = tmp.transpose((3, 2, 0, 1))
                # B, n_obj, C, H, W
            sample[elem] = torch.from_numpy(tmp)

        return sample

