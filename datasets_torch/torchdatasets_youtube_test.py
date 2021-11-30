from __future__ import division

import os
import numpy as np
import cv2

from torch.utils.data import Dataset
import json
from PIL import Image

from libs import utils_custom


class YoutubeVOS(Dataset):
    """DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""
    def __init__(self,
                 root,
                 config,
                 custom_frames=None,
                 transform=None,
                 seq_name=None,
                 resize=True,#f,h,w
                 ):
        """Loads image to label pairs for tool pose estimation
        split: Split or list of splits of the dataset
        root: dataset directory with subfolders "JPEGImages" and "Annotations"
        num_frames: Select number of frames of the sequence (None for all frames)
        custom_frames: List or Tuple with the number of the frames to include
        transform: Data transformations
        retname: Retrieve meta data in the sample key 'meta'
        seq_name: Use a specific sequence
        obj_id: Use a specific object of a sequence (If None and sequence is specified, the batch_gt is True)
        gt_only_first_frame: Provide the GT only in the first frame
        no_gt: No GT is provided
        batch_gt: For every frame sequence batch all the different objects gt
        rgb: Use RGB channel order in the image
        """
        self.imgdir_youtube = root + '/train/JPEGImages/'
        self.segdir_youtube = root + '/train/Annotations/'
        self.transform = transform
        self.seq_name = seq_name
        self.resize = resize
        self.custom_frames = custom_frames
        self.config= config

        with open(self.config.youtube_dataset_dir + '/train/meta.json') as handle:
            self.DB_youtube = json.loads(handle.read())['videos']

        # Initialize the per sequence images for online training
        names_img = np.sort(os.listdir(os.path.join(self.imgdir_youtube, str(seq_name))))
        img_list = list(map(lambda x: os.path.join(self.imgdir_youtube, str(seq_name), x), names_img))
        if custom_frames is not None:
            assert min(custom_frames) >= 0 and max(custom_frames) <= len(img_list)
            img_list = [img_list[x] for x in custom_frames]
        self.custom_frames = custom_frames
        self.img_list = img_list


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        frame_idx = self.custom_frames[idx]
        img = np.array(Image.open(self.img_list[idx]))
        if self.resize:
            img = self.resize_shorter480(img)

        pad_img, pad_info = utils_custom.apply_pad(img)

        sample = {'image': pad_img}
        sample['meta'] = {'frame_id': frame_idx,
                          'pad_size': (pad_img.shape[0], pad_img.shape[1]),
                          'pad_info': pad_info}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def resize_shorter480(self, img):
        ori_h, ori_w = img.shape[0], img.shape[1]
        if ori_w >= ori_h:
            if ori_h ==480:
                return img
            new_h = 480
            new_w = int((ori_w / ori_h) * 480)
        else:
            if ori_w ==480:
                return img
            new_w = 480
            new_h = int((ori_h / ori_w) * 480)

        output_size = (new_w, new_h)

        new_img = cv2.resize(img, output_size, interpolation=cv2.INTER_CUBIC)
        return new_img

